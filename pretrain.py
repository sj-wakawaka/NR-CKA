"""
该代码用于预训练实验
应当建立多backbone、多数据集的兼容性
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import time
from datetime import datetime
import json
import os
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from backbone.xception import FCCAXceptionModel
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.nn as nn


# 定义监督损失
def supervised_contrastive_loss(features,labels,temperature=0.07):
    """
    监督对比损失
    寻找正负样本对
    features:[N,feature_dim]
    """
    # 对比学习和余弦角度一定要先对向量做归一化
    features = F.normalize(features,dim=-1)
    # matmul操作是A@B。温度缩放依然是为了调节后续softmax值。
    logits = torch.matmul(features,features.t()) / temperature #[N,N]
    # 去掉最大值，防止后续计算有溢出
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    
    labels = labels.view(-1, 1) # -1自动判断形状大小，labels第二维度设置为1。【N，1】
    positive_mask = torch.eq(labels, labels.t()).float().to(features.device) #[N,N] 一样的为1，不一样的为0
    self_mask = torch.eye(features.size(0), device=features.device) #[N,N]，单位矩阵，对角线是1，其他事0
    positive_mask = positive_mask * (1.0 - self_mask) # 把对角线样本本身也去掉，得到每个样本同类别对的掩码矩阵
    
    # 获得有了掩码之后的logits，除了自己和自己之外的所有样本的
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    # 将logits转化为概率
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # 计算每个样本的正样本对个数
    positive_count = positive_mask.sum(dim=1).clamp_min(1.0)
    # 得到所有正样本对的logits_prob
    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count
    # 返回loss的平均值
    return loss.mean()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置日志
def set_logger(runtime,log_dir,backbone,args):   
    run_dir = os.path.join(log_dir, backbone,args.dataset,runtime)
    os.makedirs(run_dir,exist_ok=True)

    log_file = os.path.join(run_dir ,"pretrain.log")
    args_file = os.path.join(run_dir,"args.json")

    with open(args_file,"w",encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger("pretrain")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 负责把日志消息写入磁盘上的一个文件中。
    fh = logging.FileHandler(log_file, encoding="utf-8")
    # 负责把日志消息输出到“流”中
    sh = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    logger.info(f"Args file: {args_file}")
    logger.info("Training settings:")
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    return logger


def iter_window_starts(signal_len, window, step):
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if signal_len < window:
        return range(0)
    return range(0, signal_len - window + 1, step)

# 数据切片
def slice_data(length,step,x,y,logger=None, tag=""):
    if logger is not None:
        logger.info(f"{tag} slicing start: x={x.shape}, y={y.shape}")

    data = []
    label = []
    for i in range(x.shape[0]):  # x的形状是[N,C,L]
        for start in iter_window_starts(x.shape[2], length, step):
            data.append(np.expand_dims(x[i, :, start:start + length], axis=0))
            label.append(int(y[i]))
    X = np.vstack(data)
    Y = np.hstack(label)
    if logger is not None:
        logger.info(f"{tag} slicing done: X={X.shape}, Y={Y.shape}")

    return X,Y

# 获取adsb数据集
def get_adsb(path,class_nums):
    data = h5py.File(path, 'r')
    f = list(data.keys())
    x = data[f[0]][()].transpose((2, 0, 1))
    y = data[f[1]][()].squeeze()
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    indices = np.where(y<class_nums)[0]

    x_new = x[indices]
    y_new = y[indices]

    return x_new,y_new

# 获取金海豚数据集
def get_jht_files(path, class_nums, logger=None):
    half = class_nums // 2
    label_path = os.path.join(path, "label.txt")

    file_to_label = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, label = parts
            file_to_label[name] = int(label)

    selected_files = []

    for folder in ["1", "3"]:
        folder_path = os.path.join(path, folder)

        files = [
            name for name in os.listdir(folder_path)
            if name.endswith(".npy") and name in file_to_label
        ]

        files = sorted(files, key=lambda name: file_to_label[name])
        selected_files.extend([(folder, name) for name in files[:half]])

    if logger is not None:
        logger.info(f"JHT selected classes={len(selected_files)}")
        for new_label, (folder, name) in enumerate(selected_files):
            logger.info(f"JHT class {new_label}: {folder}/{name}")

    return selected_files


# 获取tx数据集
def get_tx(path, class_nums, logger=None):
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"TX data folder does not exist: {root}")

    label_files = []
    for name in os.listdir(root):
        if not name.endswith(".npy"):
            continue
        label_name = os.path.splitext(name)[0]
        if not label_name.isdigit():
            continue
        label_files.append((int(label_name), name))

    label_files = sorted(label_files, key=lambda item: item[0])
    selected_files = label_files[:class_nums]

    if len(selected_files) < class_nums:
        raise ValueError(
            f"TX expected {class_nums} class files, but found {len(selected_files)} in {root}"
        )

    raw_x_list = []
    raw_y_list = []

    if logger is not None:
        logger.info(f"TX selected classes={len(selected_files)}")

    for new_label, (source_label, name) in enumerate(selected_files):
        file_path = os.path.join(root, name)

        if logger is not None:
            logger.info(f"TX loading class={new_label}, source_label={source_label}, file={file_path}")

        x = np.asarray(np.load(file_path, mmap_mode="r"), dtype=np.float32)
        if x.ndim == 2 and x.shape[0] == 2:
            x = np.expand_dims(x, axis=0)
        elif x.ndim == 3 and x.shape[1] != 2 and x.shape[2] == 2:
            x = x.transpose((0, 2, 1))

        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"TX file {file_path} should have shape [N, 2, L], got {x.shape}")

        y = np.full((x.shape[0],), new_label, dtype=np.int64)
        raw_x_list.append(x)
        raw_y_list.append(y)

        if logger is not None:
            logger.info(f"TX loaded class={new_label}: x={x.shape}, y={y.shape}")

    train_data = np.vstack(raw_x_list).astype(np.float32)
    train_label = np.hstack(raw_y_list).astype(np.int64)

    if logger is not None:
        logger.info(f"TX selected raw data: x={train_data.shape}, y={train_label.shape}")

    return train_data, train_label

# 生成虚拟信号
def generate_virtual_signals(x,alpha_min=0.3,alpha_max=0.7,noise_std=0.05,max_shift=16,):
    """
    x_virtual = alpha * (xi + noise) + (1 - alpha) * shift(xj, delta)
    """
    batch_size = x.size(0)
    device = x.device

    perm = torch.randperm(batch_size, device=device)
    shifted_source = x[perm]
    shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,), device=device)

    shifted = torch.empty_like(shifted_source)
    for i in range(batch_size):
        shifted[i] = torch.roll(shifted_source[i], int(shifts[i].item()), dims=-1)

    alpha = torch.empty(batch_size, 1, 1, device=device).uniform_(alpha_min, alpha_max)
    noise = torch.randn_like(x) * noise_std
    return alpha * (x + noise) + (1.0 - alpha) * shifted


def add_awgn_with_random_snr(x, snr_min_db=5.0, snr_max_db=20.0):
    """Add batch-wise AWGN with random SNR in [snr_min_db, snr_max_db]."""
    if snr_min_db <= 0 or snr_max_db <= 0:
        raise ValueError("snr bounds must be positive")
    if snr_min_db > snr_max_db:
        raise ValueError("snr_min_db must be <= snr_max_db")

    batch = x.size(0)
    device = x.device
    clean_power = x.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    snr_db = torch.empty(batch, 1, 1, device=device).uniform_(snr_min_db, snr_max_db)
    noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise, snr_db.view(-1)


def normalize_array(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return ((x - x_min) / (x_max - x_min + 1e-12)).astype(np.float32)


def normalize_samples(x):
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    return ((x - x_min) / (x_max - x_min + 1e-12)).astype(np.float32)


def build_sample_id_slices(x, slice_len, step, start_sample_id=0):
    data = []
    sample_ids = []
    sample_id = start_sample_id

    for i in range(x.shape[0]):
        for start in iter_window_starts(x.shape[2], slice_len, step):
            data.append(np.expand_dims(x[i, :, start:start + slice_len], axis=0))
            sample_ids.append(sample_id)
        sample_id += 1

    return np.vstack(data), np.hstack(sample_ids), sample_id

#设置数据集
def set_dataset(dataset, data_dir, class_nums, slice_len, step, logger=None, seed=1993):
    if logger is not None:
        logger.info(
            f"building dataset={dataset}, class_nums={class_nums}, "
            f"slice_len={slice_len}, step={step}"
        )

    if dataset == "adsb":
        train_data_path = os.path.join(data_dir, "ADSB", "Task_1_Train.mat")
        test_data_path = os.path.join(data_dir, "ADSB", "Task_1_Test.mat")
        train_data, train_label = get_adsb(train_data_path, class_nums)
        test_data, test_label = get_adsb(test_data_path, class_nums)

        x_train, x_val, y_train, y_val = train_test_split(
            train_data,
            train_label,
            test_size=0.2,
            stratify=train_label,
            random_state=seed,
        )

        X_train, Y_train = slice_data(slice_len, step, x_train, y_train, logger, "ADSB train")
        X_val, Y_val = slice_data(slice_len, step, x_val, y_val, logger, "ADSB val")
        X_test, Y_test, _ = build_sample_id_slices(test_data, slice_len, step)
        X_test = normalize_array(X_test)
        test_true_label = test_label.astype(np.int64)

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

        return train_dataset, val_dataset, test_dataset, test_true_label

    if dataset == "jht":
        data_path = os.path.join(data_dir, "jht")
        selected_files = get_jht_files(data_path, class_nums, logger)

        raw_x_list = []
        raw_y_list = []

        if logger is not None:
            logger.info("JHT loading selected raw class files before normalization")

        for new_label, (folder, name) in enumerate(selected_files):
            file_path = os.path.join(data_path, folder, name)

            if logger is not None:
                logger.info(f"JHT loading class={new_label}, file={file_path}")

            x = np.load(file_path, mmap_mode="r")
            y = np.full((x.shape[0],), new_label, dtype=np.int64)

            raw_x_list.append(np.asarray(x, dtype=np.float32))
            raw_y_list.append(y)

            if logger is not None:
                logger.info(f"JHT loaded class={new_label}: x={x.shape}, y={y.shape}")

        if logger is not None:
            logger.info("JHT merging selected raw classes")

        train_data = np.vstack(raw_x_list).astype(np.float32)
        train_label = np.hstack(raw_y_list).astype(np.int64)

        if logger is not None:
            logger.info(f"JHT selected raw data: x={train_data.shape}, y={train_label.shape}")
            logger.info("JHT normalizing each raw sample before split/slicing")

        train_data = normalize_samples(train_data)

        x_train, x_temp, y_train, y_temp = train_test_split(
            train_data,
            train_label,
            test_size=0.2,
            stratify=train_label,
            random_state=seed,
            shuffle=True,
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=seed,
            shuffle=True,
        )

        if logger is not None:
            logger.info(
                f"JHT split raw samples: train={x_train.shape[0]}, "
                f"val={x_val.shape[0]}, test={x_test.shape[0]}"
            )

        X_train, Y_train = slice_data(slice_len, step, x_train, y_train, logger, "JHT train")
        X_val, Y_val = slice_data(slice_len, step, x_val, y_val, logger, "JHT val")

        X_test, Y_test, _ = build_sample_id_slices(x_test, slice_len, step)
        test_true_label = y_test.astype(np.int64)

        if logger is not None:
            logger.info(
                f"JHT sliced: X_train={X_train.shape}, Y_train={Y_train.shape}, "
                f"X_val={X_val.shape}, Y_val={Y_val.shape}, "
                f"X_test={X_test.shape}, Y_test={Y_test.shape}, "
                f"test_true_label={test_true_label.shape}"
            )

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

        return train_dataset, val_dataset, test_dataset, test_true_label

    if dataset == "tx":
        data_path = os.path.join(data_dir, "通信辐射源数据", "30_classes_merged")
        if not os.path.isdir(data_path):
            data_path = "/workspace/sunjie/work/data/通信辐射源数据/30_classes_merged"

        train_data, train_label = get_tx(data_path, class_nums, logger)

        if logger is not None:
            logger.info("TX normalizing each raw sample before split/slicing")

        train_data = normalize_samples(train_data)

        x_train, x_temp, y_train, y_temp = train_test_split(
            train_data,
            train_label,
            test_size=0.2,
            stratify=train_label,
            random_state=seed,
            shuffle=True,
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=seed,
            shuffle=True,
        )

        if logger is not None:
            logger.info(
                f"TX split raw samples: train={x_train.shape[0]}, "
                f"val={x_val.shape[0]}, test={x_test.shape[0]}"
            )

        X_train, Y_train = slice_data(slice_len, step, x_train, y_train, logger, "TX train")
        X_val, Y_val = slice_data(slice_len, step, x_val, y_val, logger, "TX val")

        X_test, Y_test, _ = build_sample_id_slices(x_test, slice_len, step)
        test_true_label = y_test.astype(np.int64)

        if logger is not None:
            logger.info(
                f"TX sliced: X_train={X_train.shape}, Y_train={Y_train.shape}, "
                f"X_val={X_val.shape}, Y_val={Y_val.shape}, "
                f"X_test={X_test.shape}, Y_test={Y_test.shape}, "
                f"test_true_label={test_true_label.shape}"
            )

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

        return train_dataset, val_dataset, test_dataset, test_true_label

    raise ValueError(f"Unknown dataset: {dataset}")


# 设置模型
def set_model(args,classes_num):
    if args.backbone ==  "xception":
        model = FCCAXceptionModel(
            feature_dim=2048,
            num_classes=classes_num,
            seq_len=args.slice_len,
            in_channels=2,
            cls_scale=args.cls_scale,
            sphere_margin=args.sphere_margin)
    return model.to(args.device)


# 预训练
def pretrain(args,logger,run_time):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    base_classes = [int(x.strip()) for x in args.base_classes.split(",") if x.strip()]
    label_to_index = {label: i for i, label in enumerate(base_classes)}

    weight_save_path = os.path.join(args.weight_dir, args.dataset, args.backbone, run_time)
    os.makedirs(weight_save_path,exist_ok=True)
    best_model_dir = os.path.join(weight_save_path,"best.pth")

    classes_num = len(base_classes)
    if args.use_virtual:
        classes_num += 1
    
    train_dataset, val_dataset, test_dataset, test_true_label = set_dataset(
        args.dataset,
        args.data_dir,
        classes_num,
        args.slice_len,
        args.step,
        logger=logger,
        seed=args.seed,
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(
        f"train_batches={len(train_dataloader)}, "
        f"val_batches={len(val_dataloader)}, "
        f"test_batches={len(test_dataloader)}"
    )

    model = set_model(args,classes_num)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val_acc = 0.0
    for epoch in range(1,args.epochs+1):
        if (epoch+1)%10 == 0:
            for g in optimizer.param_groups:
                    g['lr'] *= 0.1
        model.train()
        total_loss = 0.0
        total_rec = 0.0 # 重建损失
        total_sph = 0.0 # SphereFace损失
        total_con = 0.0 # 对比损失
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader,desc=f"Epoch {epoch}/{args.epochs}",ncols=120,leave=False)

        all_preds = []
        all_labels = []

        for batch in pbar:
            x, y = batch
            x = x.to(device).float()
            y = y.to(device).long().view(-1)
            y = torch.tensor(
                [label_to_index[int(label.item())] for label in y],
                dtype=torch.long,
                device=device,
            )

            # 如果开启了虚拟类训练
            if args.use_virtual:
                x_virtual = generate_virtual_signals(
                    x,
                    noise_std=args.noise_std,
                    max_shift=args.max_shift,
                )
                y_virtual = torch.full(
                    (x_virtual.size(0),),
                    fill_value=len(base_classes),
                    dtype=torch.long,
                    device=device,
                )
                x_all = torch.cat([x, x_virtual], dim=0)
                y_all = torch.cat([y, y_virtual], dim=0)
            else:
                x_all = x
                y_all = y

            x_noisy, snr_db = add_awgn_with_random_snr(
                x_all,
                snr_min_db=args.noise_snr_min,
                snr_max_db=args.noise_snr_max,
            )

            outputs = model(x_noisy, y_all)
            loss_rec = F.mse_loss(outputs["reconstruction"], x_all)
            loss_sph = F.cross_entropy(outputs["logits"], y_all)
            loss_con = supervised_contrastive_loss(outputs["features"],y_all,temperature=args.temperature)
            loss = (args.lambda_rec * loss_rec+ args.lambda_sph * loss_sph+ args.lambda_con * loss_con)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                eval_logits = model.classifier(outputs["features"].detach())
                pred = eval_logits.argmax(dim=1)
                correct += (pred == y_all).sum().item()
                total += y_all.numel()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y_all.cpu().numpy())

            batch_size = y_all.numel()
            total_loss += loss.item() * batch_size
            total_rec += loss_rec.item() * batch_size
            total_sph += loss_sph.item() * batch_size
            total_con += loss_con.item() * batch_size

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / max(total, 1):.4f}",
                "snr": f"{snr_db.mean().item():.1f}",
            })

        train_loss = total_loss / max(total, 1)
        train_rec = total_rec / max(total, 1)
        train_sph = total_sph / max(total, 1)
        train_con = total_con / max(total, 1)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        train_acc = accuracy_score(all_labels, all_preds)

        val_acc = evaluate(
        model=model,
        loader=val_dataloader,
        device=device,
        label_to_index=label_to_index,
    )

        

        logger.info(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"rec={train_rec:.4f} "
            f"sph={train_sph:.4f} "
            f"con={train_con:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
        )
        dir = os.path.join(weight_save_path,f"epoch_{epoch}.pth")
        torch.save({ "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict()},
                    dir)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "backbone": model.backbone.state_dict(),
                    "classifier": model.classifier.state_dict()
                },
                best_model_dir,
            )
            logger.info(f"Saved best checkpoint to {best_model_dir}")
            logger.info(f"best={best_val_acc:.4f}")
    logger.info(f"Pretraining finished. Best val_acc={best_val_acc:.4f}")
    test(test_dataloader, model, args, best_model_dir, logger, test_true_label)


# 验证代码
@torch.no_grad()
def evaluate(model, loader, device, label_to_index):
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        x, y = batch
        x = x.to(device).float()
        y = y.to(device).long().view(-1)
        y = torch.tensor(
            [label_to_index[int(label.item())] for label in y],
            dtype=torch.long,
            device=device,
        )

        outputs = model(x, labels=None)
        logits = outputs["logits"]
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)
        

# 测试
def test(test_dataloader, model, args, checkpoint_dir, logger, test_true_label):
    checkpoint = torch.load(checkpoint_dir, map_location=args.device)
    model.backbone.load_state_dict(checkpoint['backbone'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    model = model.to(args.device)
    model.eval()

    scores_labels = []
    softmax_labels = []
    sample_ids = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for x, sid in tqdm(test_dataloader):
            x = x.to(args.device).float()
            outputs = model(x)
            pred = model.classifier(outputs["features"])
            pred_softmax = softmax(pred)

            scores_labels.append(pred.cpu().numpy())
            softmax_labels.append(pred_softmax.cpu().numpy())
            sample_ids.append(sid.long().view(-1).cpu().numpy())

    scores_labels = np.vstack(scores_labels)
    softmax_labels = np.vstack(softmax_labels)
    sample_ids = np.hstack(sample_ids)
    test_true_label = np.asarray(test_true_label, dtype=np.int64)

    avg_scores_labels = []
    avg_softmax_labels = []
    vote_labels = []

    for sid in np.unique(sample_ids):
        idx = np.where(sample_ids == sid)[0]
        avg_scores_labels.append(np.argmax(np.mean(scores_labels[idx], axis=0)))
        avg_softmax_labels.append(np.argmax(np.mean(softmax_labels[idx], axis=0)))

        slice_preds = np.argmax(softmax_labels[idx], axis=1)
        vote_labels.append(np.bincount(slice_preds).argmax())

    vote_labels = np.array(vote_labels)
    avg_softmax_labels = np.array(avg_softmax_labels)
    avg_scores_labels = np.array(avg_scores_labels)

    if test_true_label.shape[0] != vote_labels.shape[0]:
        raise ValueError(
            f"test_true_label length {test_true_label.shape[0]} does not match "
            f"voted samples {vote_labels.shape[0]}"
        )

    vote_oa = accuracy_score(test_true_label, vote_labels)
    avg_scores_oa = accuracy_score(test_true_label, avg_scores_labels)
    avg_softmax_oa = accuracy_score(test_true_label, avg_softmax_labels)
    logger.info(f'投票结果：{vote_oa}, 平均分数结果：{avg_scores_oa}, 平均概率结果：{avg_softmax_oa}')



def set_parse():
    parser =  argparse.ArgumentParser()
    # 数据所在文件夹路径
    parser.add_argument("--data_dir",default="/workspace/sunjie/work/data/")

    # 数据预处理切片设置
    parser.add_argument("--slice-len", type=int, default=1000)
    parser.add_argument("--step", type=int, default=800)

    # 日志、模型存储文件夹路径
    parser.add_argument("--weight_dir",default="/workspace/sunjie/work/20260904/checkpoint/pretrain/")
    parser.add_argument("--log_dir",default="/workspace/sunjie/work/20260904/log/pretrain/")

    # 设置预训练基础类
    parser.add_argument("--base_classes",type=str,default="0,1,2,3,4,5,6,7,8,9,10")

    # 设置训练超参
    parser.add_argument("--batch_size",type=int,default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # 设置损失函数超参
    parser.add_argument("--lambda-rec", type=float, default=4.0)
    parser.add_argument("--lambda-sph", type=float, default=1.5)
    parser.add_argument("--lambda-con", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=0.07)

    # 设置sphereface超参
    parser.add_argument("--cls-scale", type=float, default=15.0)  # 温度系数
    parser.add_argument("--sphere-margin", type=int, default=2)  # 用于调节样本到类原型的距离

    # 设置数据集
    parser.add_argument("--dataset",default="jht",choices = ["adsb","jht","tx"])

    # 设置backbone
    parser.add_argument("--backbone",default="xception",choices = ["mantis","fcca","xception"])

    # 设置随机种子
    parser.add_argument("--seed",type=int,default=1993)

    parser.add_argument("--device", default="cuda")

    # 虚拟信号
    parser.add_argument("--use-virtual", action="store_true")
    parser.add_argument("--noise-snr-min", type=float, default=5.0)
    parser.add_argument("--noise-snr-max", type=float, default=20.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--max-shift", type=int, default=16)

    return parser.parse_args()



def main():
    args = set_parse()
    set_seed(args.seed)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = set_logger(run_time,args.log_dir,args.backbone,args)
    pretrain(args,logger,run_time)
    

if __name__ == "__main__":
    main()
