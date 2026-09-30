import argparse
import json
import logging
import os
import random
import sys
import math
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backbone.xception import Xception



def set_logger(runtime, log_dir, backbone, args):
    run_dir = os.path.join(log_dir, backbone, f'session_{args.session}', args.dataset, f'version_{args.version}', runtime)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "kvroute.log")
    result_file = os.path.join(run_dir, "result.log")
    args_file = os.path.join(run_dir, "args.json")

    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    open(result_file, 'a', encoding='utf-8').close()

    logger = logging.getLogger("kvroute")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    logger.info(f"Args file: {args_file}")
    logger.info("Training settings:")
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    return logger, run_dir


def log_message(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)



def remap_labels(labels, class_ids):
    class_to_local = {int(c): i for i, c in enumerate(class_ids)}
    return torch.tensor(
        [class_to_local[int(y.item())] for y in labels],
        dtype=torch.long,
        device=labels.device,
    )


class LowRankValueAdapter(nn.Module):
    """
    Task-specific low-rank value adapter.

    h' = h + alpha * Up(Down(h))
    """

    def __init__(self, feature_dim=2048, rank=8, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return h + self.alpha * self.up(self.down(h))


class TaskKVAdapter(nn.Module):
    """
    每个 task 一个 adapter:

    - class_keys: [m, D]
    - route_encoder / route_decoder: task subspace autoencoder
    - class_value_adapters: task内每个class的值修正器
    - task-specific classifier
    """

    def __init__(
        self,
        feature_dim=2048,
        num_classes_per_task=5,
        rank=8,
        route_rank=16,
        adapter_alpha=1.0,
        key_prototypes=3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes_per_task = num_classes_per_task
        self.route_rank = route_rank
        self.key_prototypes = key_prototypes

        self.class_keys = nn.Parameter(torch.randn(num_classes_per_task, key_prototypes, feature_dim))

        self.route_encoder = nn.Linear(feature_dim, route_rank, bias=False)
        self.route_decoder = nn.Linear(route_rank, feature_dim, bias=False)

        nn.init.orthogonal_(self.route_encoder.weight)
        with torch.no_grad():
            self.route_decoder.weight.copy_(self.route_encoder.weight.t())

        self.class_value_adapters = nn.ModuleList(
            [LowRankValueAdapter(feature_dim=feature_dim, rank=rank, alpha=adapter_alpha)
             for _ in range(num_classes_per_task)]
        )

        self.classifier = nn.Linear(feature_dim, num_classes_per_task)


    def class_key_scores(self, h):
        h = F.normalize(h, dim=-1)
        keys = F.normalize(self.class_keys, dim=-1)
        flat_keys = keys.reshape(self.num_classes_per_task * self.key_prototypes, self.feature_dim)
        scores = h @ flat_keys.t()
        scores = scores.reshape(h.size(0), self.num_classes_per_task, self.key_prototypes)
        return scores.max(dim=2).values






    def reconstruction_error(self, h):
        h_norm = F.normalize(h, dim=-1)
        z = self.route_encoder(h_norm)
        h_hat = self.route_decoder(z)
        h_hat = F.normalize(h_hat, dim=-1)
        return (h_norm - h_hat).pow(2).sum(dim=-1)


    def route_reconstruction_loss(self, h):
        return self.reconstruction_error(h).mean()

    def reconstruction_loss(self, h):
        return self.route_reconstruction_loss(h)


    def route_parameters(self):
        return [
            self.class_keys,
            *self.route_encoder.parameters(),
            *self.route_decoder.parameters(),
            *self.class_value_adapters.parameters(),
            *self.classifier.parameters(),
        ]

    def class_logits(self, h):
        corrected = torch.stack([adapter(h) for adapter in self.class_value_adapters], dim=1)
        bsz, num_classes, feat_dim = corrected.shape
        flat_logits = self.classifier(corrected.reshape(bsz * num_classes, feat_dim)).reshape(
            bsz, num_classes, num_classes
        )
        diag_idx = torch.arange(num_classes, device=h.device)
        local_logits = flat_logits[:, diag_idx, diag_idx]
        return local_logits, corrected

    def forward(self, h):
        logits, corrected = self.class_logits(h)
        return logits, corrected




class KVRouteXception(nn.Module):
    def __init__(
        self,
        backbone,
        feature_dim=2048,
        num_classes=100,
        rank=8,
        top_k_tasks=3,
        adapter_alpha=1.0,
        route_rank=16,
        class_route_tau=0.07,
        normalize_backbone=True,
        key_prototypes=3,
    ):
        super().__init__()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.rank = rank
        self.top_k_tasks = top_k_tasks
        self.adapter_alpha = adapter_alpha
        self.route_rank = route_rank
        self.class_route_tau = class_route_tau
        self.normalize_backbone = normalize_backbone
        self.key_prototypes = key_prototypes

        self.adapters = nn.ModuleDict()
        self.task_classes = {}
        self.task_ids = []

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def add_task(self, task_id, class_ids):
        class_ids = list(class_ids)

        if task_id not in self.task_ids:
            self.task_ids.append(task_id)

        self.task_classes[task_id] = class_ids

        adapter = TaskKVAdapter(
            feature_dim=self.feature_dim,
            num_classes_per_task=len(class_ids),
            rank=self.rank,
            route_rank=self.route_rank,
            adapter_alpha=self.adapter_alpha,
            key_prototypes=self.key_prototypes,
        )
        adapter = adapter.to(next(self.backbone.parameters()).device)
        self.adapters[str(task_id)] = adapter

    def freeze_all_tasks(self):
        for adapter in self.adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False


    def unfreeze_route_task(self, task_id):
        adapter = self.adapters[str(task_id)]
        for p in adapter.parameters():
            p.requires_grad = False
        for p in adapter.route_parameters():
            p.requires_grad = True


    def route_trainable_task_parameters(self, task_id):
        return self.adapters[str(task_id)].route_parameters()

    def extract_backbone(self, x):
        with torch.no_grad():
            h = self.backbone(x)

        if self.normalize_backbone:
            h = F.normalize(h, dim=-1)

        return h

    def route_stage1_class_keys(
        self,
        h,
        seen_task_ids,
        top_k_tasks=3,
    ):
        """
        第一阶段：
        对每个 task 的全部 class-key 直接打分，用 task 内最大 class-key 相似度作为 task 分数。
        """

        task_scores = []

        for task_id in seen_task_ids:
            adapter = self.adapters[str(task_id)]
            class_scores = adapter.class_key_scores(h)
            score = class_scores.max(dim=1).values
            task_scores.append(score)

        stage1_scores = torch.stack(task_scores, dim=1)

        k = min(top_k_tasks, len(seen_task_ids))
        top_task_pos = torch.topk(stage1_scores, k=k, dim=1).indices

        return stage1_scores, top_task_pos


    def route(
        self,
        h,
        seen_task_ids,
        top_k_tasks=None,
    ):
        if top_k_tasks is None:
            top_k_tasks = self.top_k_tasks

        stage1_scores, top_task_pos = self.route_stage1_class_keys(
            h=h,
            seen_task_ids=seen_task_ids,
            top_k_tasks=top_k_tasks,
        )

        return stage1_scores, stage1_scores, top_task_pos

    def calibrate_logits(self, logits, mode="zscore"):
        if mode == "none":
            return logits

        if mode == "center":
            return logits - logits.mean(dim=1, keepdim=True)

        if mode == "zscore":
            if logits.size(1) <= 1:
                return logits - logits.mean(dim=1, keepdim=True)
            mean = logits.mean(dim=1, keepdim=True)
            std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (logits - mean) / std

        if mode == "l2":
            return F.normalize(logits, dim=1)

        raise ValueError(f"Unknown logit calibration mode: {mode}")

    def forward(
        self,
        x,
        seen_task_ids,
        top_k_tasks=None,
        logit_calibration="zscore",
    ):
        h = self.extract_backbone(x)

        route_scores, stage1_scores, top_task_pos = self.route(
            h=h,
            seen_task_ids=seen_task_ids,
            top_k_tasks=top_k_tasks,
        )

        selected_task_pos = route_scores.argmax(dim=1)

        global_logits = h.new_full(
            (h.size(0), self.num_classes),
            -1e9,
        )

        for local_pos, task_id in enumerate(seen_task_ids):
            mask = selected_task_pos == local_pos

            if not mask.any():
                continue

            adapter = self.adapters[str(task_id)]
            logits_local, _ = adapter(h[mask])
            logits_local = self.calibrate_logits(
                logits_local,
                mode=logit_calibration,
            )

            class_ids = torch.tensor(
                self.task_classes[task_id],
                dtype=torch.long,
                device=h.device,
            )

            row_ids = mask.nonzero(as_tuple=False).view(-1)

            global_logits[
                row_ids.unsqueeze(1),
                class_ids.unsqueeze(0),
            ] = logits_local

        return {
            "features": h,
            "logits": global_logits,
            "route_scores": route_scores,
            "stage1_scores": stage1_scores,
            "top_task_pos": top_task_pos,
        }




def stage1_classkey_losses(
    model,
    h,
    y,
    task_id,
    seen_task_ids,
    tau=0.07,
    margin=0.1,
):
    adapter = model.adapters[str(task_id)]
    class_ids = model.task_classes[task_id]
    local_targets = torch.empty_like(y)
    for local_id, class_id in enumerate(class_ids):
        local_targets[y == int(class_id)] = int(local_id)

    local_scores = adapter.class_key_scores(h)
    local_ce = F.cross_entropy(local_scores / tau, local_targets)

    pos_scores = local_scores.max(dim=1).values
    neg_scores = []
    for old_id in seen_task_ids:
        if old_id == task_id:
            continue
        old_adapter = model.adapters[str(old_id)]
        neg_scores.append(old_adapter.class_key_scores(h).max(dim=1).values)

    if neg_scores:
        hardest_neg = torch.stack(neg_scores, dim=1).max(dim=1).values
        margin_loss = F.relu(margin - pos_scores + hardest_neg).mean()
    else:
        margin_loss = h.new_tensor(0.0)

    return local_ce, margin_loss


def classkey_inter_task_separation_loss(
    model,
    task_id,
    seen_task_ids,
    margin=0.1,
):
    if len(seen_task_ids) <= 1:
        return next(model.parameters()).new_tensor(0.0)

    adapter = model.adapters[str(task_id)]
    cur_keys = F.normalize(adapter.class_keys.reshape(-1, model.feature_dim), dim=-1)

    old_keys = []
    for old_id in seen_task_ids:
        if old_id == task_id:
            continue
        old_keys.append(model.adapters[str(old_id)].class_keys.detach().reshape(-1, model.feature_dim))

    if not old_keys:
        return cur_keys.new_tensor(0.0)

    old_keys = F.normalize(torch.cat(old_keys, dim=0), dim=-1)
    sim = cur_keys @ old_keys.t()
    return F.relu(sim - margin).pow(2).mean()



def add_awgn_with_fixed_snr(x, snr_db):
    if snr_db <= 0:
        raise ValueError('snr_db must be positive')

    clean_power = x.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    snr_tensor = x.new_full((x.size(0), 1, 1), float(snr_db))
    noise_power = clean_power / (10.0 ** (snr_tensor / 10.0))
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def reconstruction_classkey_negative_loss(
    model,
    h,
    task_id,
    seen_task_ids,
    margin=0.1,
):
    if len(seen_task_ids) <= 1:
        return h.new_tensor(0.0)

    adapter = model.adapters[str(task_id)]
    pos_error = adapter.reconstruction_error(h).mean()

    neg_keys = []
    for old_id in seen_task_ids:
        if old_id == task_id:
            continue
        old_adapter = model.adapters[str(old_id)]
        neg_keys.append(old_adapter.class_keys.detach().reshape(-1, model.feature_dim))

    if not neg_keys:
        return h.new_tensor(0.0)

    neg_keys = torch.cat(neg_keys, dim=0)
    neg_error = adapter.reconstruction_error(neg_keys)
    hardest_neg_error = neg_error.min()
    return F.relu(margin + pos_error - hardest_neg_error)


def train_one_task_kvroute(
    model,
    loader,
    task_id,
    seen_task_ids,
    optimizer,
    device,
    epochs=50,
    lambda_class_local=1.0,
    lambda_stage1_margin=1.0,
    lambda_classkey_sep=0.0,
    top_k_tasks=3,
    logger=None,
):
    route_target = seen_task_ids.index(task_id)
    logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()

        total = 0
        correct_stage1 = 0
        correct_classifier = 0
        loss_sum = 0.0
        stage1_sum = 0.0
        classifier_sum = 0.0
        class_local_sum = 0.0
        stage1_margin_sum = 0.0
        classkey_sep_sum = 0.0

        for batch in loader:
            x, y = batch[:2]
            x = x.to(device).float()
            y = y.to(device).long().view(-1)

            h = model.extract_backbone(x)
            adapter = model.adapters[str(task_id)]

            route_scores, stage1_scores, _ = model.route(
                h=h,
                seen_task_ids=seen_task_ids,
                top_k_tasks=len(seen_task_ids),
            )

            route_targets = torch.full(
                (y.size(0),),
                route_target,
                dtype=torch.long,
                device=device,
            )

            stage1_logits = stage1_scores / model.class_route_tau
            loss_stage1 = F.cross_entropy(stage1_logits, route_targets)
            loss_class_local, loss_stage1_margin = stage1_classkey_losses(
                model=model,
                h=h,
                y=y,
                task_id=task_id,
                seen_task_ids=seen_task_ids,
                tau=model.class_route_tau,
                margin=0.1,
            )
            local_targets = remap_labels(y, model.task_classes[task_id])
            classifier_logits, _ = adapter(h)
            loss_classifier = F.cross_entropy(classifier_logits, local_targets)

            loss_classkey_sep = classkey_inter_task_separation_loss(
                model=model,
                task_id=task_id,
                seen_task_ids=seen_task_ids,
                margin=0.1,
            )
            loss = (
                loss_stage1
                + loss_classifier
                + lambda_class_local * loss_class_local
                + lambda_stage1_margin * loss_stage1_margin
                + lambda_classkey_sep * loss_classkey_sep
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_stage1 = stage1_logits.argmax(dim=1)
            pred_classifier = classifier_logits.argmax(dim=1)
            correct_stage1 += (pred_stage1 == route_targets).sum().item()
            correct_classifier += (pred_classifier == local_targets).sum().item()
            total += y.numel()

            loss_sum += loss.item() * y.numel()
            stage1_sum += loss_stage1.item() * y.numel()
            classifier_sum += loss_classifier.item() * y.numel()
            class_local_sum += loss_class_local.item() * y.numel()
            stage1_margin_sum += loss_stage1_margin.item() * y.numel()
            classkey_sep_sum += loss_classkey_sep.item() * y.numel()

        log = {
            "epoch": epoch,
            "loss": loss_sum / max(total, 1),
            "stage1_loss": stage1_sum / max(total, 1),
            "classifier_loss": classifier_sum / max(total, 1),
            "class_local_loss": class_local_sum / max(total, 1),
            "stage1_margin_loss": stage1_margin_sum / max(total, 1),
            "classkey_sep_loss": classkey_sep_sum / max(total, 1),
            "stage1_train_acc": correct_stage1 / max(total, 1),
            "classifier_train_acc": correct_classifier / max(total, 1),
        }
        logs.append(log)

        log_message(
            logger,
            f"epoch={epoch:03d} "
            f"loss={log['loss']:.4f} "
            f"s1_loss={log['stage1_loss']:.4f} "
            f"cls_loss={log['classifier_loss']:.4f} "
            f"class_local={log['class_local_loss']:.4f} "
            f"s1_margin={log['stage1_margin_loss']:.4f} "
            f"key_sep={log['classkey_sep_loss']:.4f} "
            f"train_s1_acc={log['stage1_train_acc']:.4f} "
            f"train_cls_acc={log['classifier_train_acc']:.4f}",
        )

    return logs




@torch.no_grad()
def initialize_task_keys(
    model,
    loader,
    task_id,
    class_ids,
    device,
):
    """
    class key 用每个类的特征均值初始化。
    """

    model.eval()
    all_features = []
    all_labels = []

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device).float()
        y = y.to(device).long().view(-1)
        h = model.extract_backbone(x)
        all_features.append(h.cpu())
        all_labels.append(y.cpu())

    features = torch.cat(all_features, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)
    adapter = model.adapters[str(task_id)]

    class_keys = []
    key_prototypes = adapter.key_prototypes
    for class_id in class_ids:
        mask = labels == int(class_id)
        feat_c = F.normalize(features[mask], dim=-1)

        if feat_c.size(0) == 0:
            keys = torch.randn(key_prototypes, model.feature_dim, device=device)
            keys = F.normalize(keys, dim=-1)
        elif feat_c.size(0) < key_prototypes:
            mean_key = F.normalize(feat_c.mean(dim=0, keepdim=True), dim=-1)
            keys = mean_key.expand(key_prototypes, -1).clone()
        else:
            mean_key = F.normalize(feat_c.mean(dim=0, keepdim=True), dim=-1)
            dist = 1.0 - (feat_c @ mean_key.t()).squeeze(1)
            centers = [mean_key.squeeze(0)]
            farthest_idx = dist.argmax()
            centers.append(feat_c[farthest_idx])

            if key_prototypes > 2:
                while len(centers) < key_prototypes:
                    center_stack = F.normalize(torch.stack(centers, dim=0), dim=-1)
                    sim_to_centers = feat_c @ center_stack.t()
                    next_idx = sim_to_centers.max(dim=1).values.argmin()
                    centers.append(feat_c[next_idx])

            centers = F.normalize(torch.stack(centers[:key_prototypes], dim=0), dim=-1)
            for _ in range(5):
                sim = feat_c @ centers.t()
                assign = sim.argmax(dim=1)
                updated = []
                for proto_id in range(key_prototypes):
                    proto_feat = feat_c[assign == proto_id]
                    if proto_feat.numel() == 0:
                        updated.append(centers[proto_id])
                    else:
                        updated.append(proto_feat.mean(dim=0))
                centers = F.normalize(torch.stack(updated, dim=0), dim=-1)
            keys = centers

        class_keys.append(keys)

    class_keys = torch.stack(class_keys, dim=0)
    adapter.class_keys.data.copy_(class_keys)





@torch.no_grad()
def majority_sample_route_predictions(route_scores, sample_ids):
    preds = route_scores.detach().cpu().argmax(dim=1).numpy()
    sample_ids_np = sample_ids.detach().cpu().numpy()

    sample_route_preds = {}
    for sid in np.unique(sample_ids_np):
        idx = np.where(sample_ids_np == sid)[0]
        sample_route_preds[int(sid)] = int(np.bincount(preds[idx]).argmax())

    return sample_route_preds




def average_incremental_task_vote(task_metrics, base_task_id=None):
    if not task_metrics:
        return 0.0

    incremental_metrics = [
        m["vote_acc"]
        for task_id, m in task_metrics.items()
        if base_task_id is None or int(task_id) != int(base_task_id)
    ]
    if incremental_metrics:
        return float(np.mean(incremental_metrics))

    return float(task_metrics.get(base_task_id, next(iter(task_metrics.values())))["vote_acc"])


@torch.no_grad()
def evaluate_route_vote_metrics(route_scores, route_labels, sample_ids, seen_task_ids):
    route_labels = route_labels.detach().cpu().numpy()
    sample_ids_np = sample_ids.detach().cpu().numpy()
    sample_route_preds = majority_sample_route_predictions(route_scores, sample_ids)

    sample_preds = []
    sample_labels = []

    for sid in np.unique(sample_ids_np):
        idx = np.where(sample_ids_np == sid)[0]
        pred_pos = sample_route_preds[int(sid)]
        pred_task = int(seen_task_ids[pred_pos])
        sample_preds.append(pred_task)
        sample_labels.append(int(route_labels[idx][0]))

    sample_preds = np.array(sample_preds)
    sample_labels = np.array(sample_labels)

    task_metrics = {}
    for task_id in sorted(np.unique(route_labels).tolist()):
        task_mask = route_labels == task_id
        task_sample_ids = sample_ids_np[task_mask]
        task_preds = []
        task_labels = []

        for sid in np.unique(task_sample_ids):
            idx = np.where(sample_ids_np == sid)[0]
            pred_pos = sample_route_preds[int(sid)]
            pred_task = int(seen_task_ids[pred_pos])
            task_preds.append(pred_task)
            task_labels.append(int(route_labels[idx][0]))

        task_preds = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_metrics[int(task_id)] = {
            "vote_acc": (task_preds == task_labels).mean().item() if task_labels.size else 0.0,
            "num_samples": int(np.unique(task_sample_ids).shape[0]),
        }

    avg_task_vote_acc = average_incremental_task_vote(task_metrics, base_task_id=None)
    last_task_id = max(task_metrics.keys()) if task_metrics else None
    last_task_vote_acc = task_metrics[last_task_id]["vote_acc"] if last_task_id is not None else 0.0

    return {
        "route_vote_acc": (sample_preds == sample_labels).mean().item() if sample_labels.size else 0.0,
        "avg_task_vote_acc": avg_task_vote_acc,
        "last_task_vote_acc": last_task_vote_acc,
        "task_metrics": task_metrics,
    }


@torch.no_grad()
def majority_sample_class_predictions(logits, sample_ids):
    preds = logits.detach().cpu().argmax(dim=1).numpy()
    sample_ids_np = sample_ids.detach().cpu().numpy()

    sample_preds = {}
    for sid in np.unique(sample_ids_np):
        idx = np.where(sample_ids_np == sid)[0]
        sample_preds[int(sid)] = int(np.bincount(preds[idx]).argmax())

    return sample_preds


@torch.no_grad()
def evaluate_class_vote_metrics(logits, labels, route_labels, sample_ids):
    labels_np = labels.detach().cpu().numpy()
    route_labels_np = route_labels.detach().cpu().numpy()
    sample_ids_np = sample_ids.detach().cpu().numpy()
    sample_preds_by_id = majority_sample_class_predictions(logits, sample_ids)

    sample_preds = []
    sample_labels = []
    sample_route_labels = []
    for sid in np.unique(sample_ids_np):
        idx = np.where(sample_ids_np == sid)[0]
        sample_preds.append(sample_preds_by_id[int(sid)])
        sample_labels.append(int(labels_np[idx][0]))
        sample_route_labels.append(int(route_labels_np[idx][0]))

    sample_preds = np.array(sample_preds)
    sample_labels = np.array(sample_labels)
    sample_route_labels = np.array(sample_route_labels)

    class_metrics = {}
    for class_id in sorted(np.unique(sample_labels).tolist()):
        mask = sample_labels == class_id
        class_metrics[int(class_id)] = {
            "vote_acc": (sample_preds[mask] == sample_labels[mask]).mean().item() if mask.any() else 0.0,
            "num_samples": int(mask.sum()),
        }

    task_metrics = {}
    for task_id in sorted(np.unique(sample_route_labels).tolist()):
        mask = sample_route_labels == task_id
        task_metrics[int(task_id)] = {
            "vote_acc": (sample_preds[mask] == sample_labels[mask]).mean().item() if mask.any() else 0.0,
            "num_samples": int(mask.sum()),
        }

    avg_task_class_vote_acc = float(np.mean([m["vote_acc"] for m in task_metrics.values()])) if task_metrics else 0.0
    last_task_id = max(task_metrics.keys()) if task_metrics else None
    last_task_class_vote_acc = task_metrics[last_task_id]["vote_acc"] if last_task_id is not None else 0.0

    return {
        "class_vote_acc": (sample_preds == sample_labels).mean().item() if sample_labels.size else 0.0,
        "avg_task_class_vote_acc": avg_task_class_vote_acc,
        "last_task_class_vote_acc": last_task_class_vote_acc,
        "task_metrics": task_metrics,
        "class_metrics": class_metrics,
    }


@torch.no_grad()
def evaluate_stage1_route_class_metrics(
    model,
    loader,
    seen_task_ids,
    device,
    top_k_tasks=3,
    logit_calibration="zscore",
    noise_snr_db=None,
):
    model.eval()

    class_to_task = {}
    for task_id in seen_task_ids:
        for class_id in model.task_classes[task_id]:
            class_to_task[int(class_id)] = task_id

    all_sample_ids = []
    all_labels = []
    all_route_labels = []
    all_stage1_scores = []
    all_logits = []

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device).float()
        y = y.to(device).long().view(-1)

        if noise_snr_db is not None:
            x = add_awgn_with_fixed_snr(x, noise_snr_db)

        output = model(
            x=x,
            seen_task_ids=seen_task_ids,
            top_k_tasks=top_k_tasks,
            logit_calibration=logit_calibration,
        )

        route_labels = torch.tensor(
            [class_to_task[int(label.item())] for label in y],
            dtype=torch.long,
            device=device,
        )

        sample_ids = batch[2].long().view(-1).cpu()
        all_sample_ids.append(sample_ids)
        all_labels.append(y.cpu())
        all_route_labels.append(route_labels.cpu())
        all_stage1_scores.append(output["stage1_scores"].cpu())
        all_logits.append(output["logits"].cpu())

    all_sample_ids = torch.cat(all_sample_ids)
    all_labels = torch.cat(all_labels)
    all_route_labels = torch.cat(all_route_labels)
    all_stage1_scores = torch.cat(all_stage1_scores)
    all_logits = torch.cat(all_logits)

    route_metrics = evaluate_route_vote_metrics(
        all_stage1_scores,
        all_route_labels,
        all_sample_ids,
        seen_task_ids,
    )
    class_metrics = evaluate_class_vote_metrics(
        all_logits,
        all_labels,
        all_route_labels,
        all_sample_ids,
    )

    return {
        "route": route_metrics,
        "class": class_metrics,
    }




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_array(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return ((x - x_min) / (x_max - x_min + 1e-12)).astype(np.float32)


def normalize_samples(x):
    x_min = np.min(x, axis=(1, 2), keepdims=True)
    x_max = np.max(x, axis=(1, 2), keepdims=True)
    return ((x - x_min) / (x_max - x_min + 1e-12)).astype(np.float32)


def iter_window_starts(signal_len, window, step):
    if window <= 0 or step <= 0:
        raise ValueError('window and step must be positive')
    if signal_len < window:
        return range(0)
    return range(0, signal_len - window + 1, step)


def slice_data(length, step, x, y, with_sample_id=False, start_sample_id=0):
    data = []
    labels = []
    sample_ids = []
    sample_id = start_sample_id

    for i in range(x.shape[0]):
        for start in iter_window_starts(x.shape[2], length, step):
            data.append(np.expand_dims(x[i, :, start:start + length], axis=0))
            labels.append(int(y[i]))
            if with_sample_id:
                sample_ids.append(sample_id)
        sample_id += 1

    X = np.vstack(data).astype(np.float32)
    Y = np.hstack(labels).astype(np.int64)
    if not with_sample_id:
        return X, Y

    S = np.hstack(sample_ids).astype(np.int64)
    return X, Y, S, sample_id


def get_adsb(path, class_nums):
    data = h5py.File(path, 'r')
    keys = list(data.keys())
    x = data[keys[0]][()].transpose((2, 0, 1))
    y = data[keys[1]][()].squeeze().astype(np.int64)
    x = normalize_array(x)

    indices = np.where(y < class_nums)[0]
    return x[indices], y[indices]


def read_jht_label_file(path):
    label_path = os.path.join(path, 'label.txt')
    label_to_file = {}

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, label = parts
            label_to_file[int(label)] = name

    return label_to_file


def infer_jht_folder(path, name):
    if '_iq_1_' in name:
        return '1'
    if '_iq_3_' in name:
        return '3'
    return '1' if os.path.exists(os.path.join(path, '1', name)) else '3'


def build_jht_local_to_raw(path, base_classes, num_classes, logger=None):
    if base_classes % 2 != 0 or num_classes % 2 != 0:
        raise ValueError('JHT base_classes and num_classes should be even.')
    if num_classes < base_classes:
        raise ValueError('num_classes must be >= base_classes.')

    label_to_file = read_jht_label_file(path)
    folder1_labels = sorted(
        label for label, name in label_to_file.items()
        if infer_jht_folder(path, name) == '1'
    )
    folder3_labels = sorted(
        label for label, name in label_to_file.items()
        if infer_jht_folder(path, name) == '3'
    )

    base_half = base_classes // 2
    total_half = num_classes // 2
    raw_order = (
        folder1_labels[:base_half]
        + folder3_labels[:base_half]
        + folder1_labels[base_half:total_half]
        + folder3_labels[base_half:total_half]
    )

    if len(raw_order) != num_classes:
        raise ValueError(
            f'JHT only found {len(raw_order)} labels for num_classes={num_classes}. '
            f'folder1={len(folder1_labels)}, folder3={len(folder3_labels)}'
        )

    local_to_raw = {local_label: raw_label for local_label, raw_label in enumerate(raw_order)}

    if logger is not None:
        pretrained_raw = [local_to_raw[i] for i in range(base_classes)]
        incremental_raw = [local_to_raw[i] for i in range(base_classes, num_classes)]
        log_message(logger, f'JHT pretrained local labels 0..{base_classes - 1} map to raw labels {pretrained_raw}')
        log_message(logger, f'JHT incremental local labels {base_classes}..{num_classes - 1} map to raw labels {incremental_raw}')

    return local_to_raw, label_to_file


def get_jht_files(path, class_ids, local_to_raw, label_to_file, logger=None):
    selected_files = []
    for local_label in class_ids:
        local_label = int(local_label)
        if local_label not in local_to_raw:
            raise ValueError(f'JHT local label {local_label} not found in local_to_raw mapping')

        raw_label = int(local_to_raw[local_label])
        if raw_label not in label_to_file:
            raise ValueError(f'JHT raw label {raw_label} not found in label.txt')

        name = label_to_file[raw_label]
        folder = infer_jht_folder(path, name)
        selected_files.append((local_label, raw_label, folder, name))

    log_message(logger, f'JHT selected local classes={list(class_ids)}')
    for local_label, raw_label, folder, name in selected_files:
        log_message(logger, f'JHT local {local_label} -> raw {raw_label}: {folder}/{name}')

    return selected_files


def get_tx_files(path, class_ids, logger=None):
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f'TX data folder does not exist: {root}')

    raw_labels = []
    for name in os.listdir(root):
        if not name.endswith('.npy'):
            continue
        label_name = os.path.splitext(name)[0]
        if label_name.isdigit():
            raw_labels.append(int(label_name))

    raw_labels = sorted(raw_labels)
    if len(raw_labels) <= max(class_ids):
        raise ValueError(
            f'TX expected at least {max(class_ids) + 1} class files, '
            f'but found {len(raw_labels)} in {root}'
        )

    selected_files = []
    for local_label in class_ids:
        raw_label = raw_labels[int(local_label)]
        selected_files.append((int(local_label), raw_label, f'{raw_label}.npy'))

    log_message(logger, f'TX selected local classes={list(class_ids)}')
    raw_order = [raw_labels[i] for i in range(max(class_ids) + 1)]
    log_message(logger, f'TX local labels map to raw labels {raw_order}')
    for local_label, raw_label, name in selected_files:
        log_message(logger, f'TX local {local_label} -> raw {raw_label}: {name}')

    return selected_files


def load_tx_array(file_path):
    x = np.asarray(np.load(file_path, mmap_mode='r'), dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 2:
        x = np.expand_dims(x, axis=0)
    elif x.ndim == 3 and x.shape[1] != 2 and x.shape[2] == 2:
        x = x.transpose((0, 2, 1))

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f'TX file {file_path} should have shape [N, 2, L], got {x.shape}')

    return x




def set_dataset(args, logger):
    log_message(
        logger,
        f'building dataset={args.dataset}, num_classes={args.num_classes}, '
        f'slice_len={args.slice_len}, step={args.step}',
    )

    if args.dataset == 'adsb':
        train_path = os.path.join(args.data_dir, 'ADSB', 'Task_1_Train.mat')
        test_path = os.path.join(args.data_dir, 'ADSB', 'Task_1_Test.mat')
        train_x, train_y = get_adsb(train_path, args.num_classes)
        test_x, test_y = get_adsb(test_path, args.num_classes)

        x_train, x_val, y_train, y_val = train_test_split(
            train_x,
            train_y,
            test_size=args.val_ratio,
            stratify=train_y,
            random_state=args.seed,
            shuffle=True,
        )

        log_message(logger, f'ADSB raw: train={train_x.shape}, test={test_x.shape}')
        log_message(logger, f'ADSB split: train={x_train.shape[0]}, val={x_val.shape[0]}, test={test_x.shape[0]}')

    elif args.dataset == 'jht':
        data_path = os.path.join(args.data_dir, 'jht')
        class_ids = list(range(args.num_classes))
        local_to_raw, label_to_file = build_jht_local_to_raw(
            data_path,
            args.base_classes,
            args.num_classes,
            logger,
        )
        selected_files = get_jht_files(data_path, class_ids, local_to_raw, label_to_file, logger)
        raw_x_list = []
        raw_y_list = []

        log_message(logger, 'JHT loading selected raw class files before normalization')
        for class_id, raw_label, folder, name in selected_files:
            file_path = os.path.join(data_path, folder, name)
            log_message(logger, f'JHT loading local={class_id}, raw={raw_label}, file={file_path}')
            x = np.load(file_path, mmap_mode='r')
            y = np.full((x.shape[0],), class_id, dtype=np.int64)
            raw_x_list.append(np.asarray(x, dtype=np.float32))
            raw_y_list.append(y)
            log_message(logger, f'JHT loaded {name}: x={x.shape}, y={y.shape}, dtype={x.dtype}')

        log_message(logger, 'JHT merging selected raw classes')
        raw_x = np.vstack(raw_x_list).astype(np.float32)
        raw_y = np.hstack(raw_y_list).astype(np.int64)
        log_message(logger, f'JHT selected raw data: x={raw_x.shape}, y={raw_y.shape}')
        log_message(logger, 'JHT normalizing each raw sample before split/slicing')
        raw_x = normalize_samples(raw_x)

        x_train, x_temp, y_train, y_temp = train_test_split(
            raw_x,
            raw_y,
            test_size=args.val_ratio + args.test_ratio,
            stratify=raw_y,
            random_state=args.seed,
            shuffle=True,
        )
        relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
        x_val, test_x, y_val, test_y = train_test_split(
            x_temp,
            y_temp,
            test_size=relative_test_ratio,
            stratify=y_temp,
            random_state=args.seed,
            shuffle=True,
        )

        log_message(logger, f'JHT split raw samples: train={x_train.shape[0]}, val={x_val.shape[0]}, test={test_x.shape[0]}')

    elif args.dataset == 'tx':
        data_path = os.path.join(args.data_dir, '通信辐射源数据', '30_classes_merged')
        if not os.path.isdir(data_path):
            data_path = '/workspace/sunjie/work/data/通信辐射源数据/30_classes_merged'

        class_ids = list(range(args.num_classes))
        selected_files = get_tx_files(data_path, class_ids, logger)
        raw_x_list = []
        raw_y_list = []

        log_message(logger, 'TX loading selected raw class files before normalization')
        for class_id, raw_label, name in selected_files:
            file_path = os.path.join(data_path, name)
            log_message(logger, f'TX loading local={class_id}, raw={raw_label}, file={file_path}')
            x = load_tx_array(file_path)
            y = np.full((x.shape[0],), class_id, dtype=np.int64)
            raw_x_list.append(x)
            raw_y_list.append(y)
            log_message(logger, f'TX loaded {name}: x={x.shape}, y={y.shape}, dtype={x.dtype}')

        log_message(logger, 'TX merging selected raw classes')
        raw_x = np.vstack(raw_x_list).astype(np.float32)
        raw_y = np.hstack(raw_y_list).astype(np.int64)
        log_message(logger, f'TX selected raw data: x={raw_x.shape}, y={raw_y.shape}')
        log_message(logger, 'TX normalizing each raw sample before split/slicing')
        raw_x = normalize_samples(raw_x)

        x_train, x_temp, y_train, y_temp = train_test_split(
            raw_x,
            raw_y,
            test_size=args.val_ratio + args.test_ratio,
            stratify=raw_y,
            random_state=args.seed,
            shuffle=True,
        )
        relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
        x_val, test_x, y_val, test_y = train_test_split(
            x_temp,
            y_temp,
            test_size=relative_test_ratio,
            stratify=y_temp,
            random_state=args.seed,
            shuffle=True,
        )

        log_message(logger, f'TX split raw samples: train={x_train.shape[0]}, val={x_val.shape[0]}, test={test_x.shape[0]}')

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    X_train, Y_train = slice_data(args.slice_len, args.step, x_train, y_train)
    X_val, Y_val, S_val, _ = slice_data(args.slice_len, args.step, x_val, y_val, with_sample_id=True)
    X_test, Y_test, S_test, _ = slice_data(args.slice_len, args.step, test_x, test_y, with_sample_id=True)

    log_message(
        logger,
        f'sliced: X_train={X_train.shape}, Y_train={Y_train.shape}, '
        f'X_val={X_val.shape}, Y_val={Y_val.shape}, S_val={S_val.shape}, '
        f'X_test={X_test.shape}, Y_test={Y_test.shape}, S_test={S_test.shape}',
    )

    train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    val_set = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long(), torch.from_numpy(S_val).long())
    test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long(), torch.from_numpy(S_test).long())
    return train_set, val_set, test_set


def select_classes(dataset, class_ids):
    labels = dataset.tensors[1].numpy()
    class_ids = set(int(c) for c in class_ids)
    indices = [i for i, y in enumerate(labels) if int(y) in class_ids]
    return Subset(dataset, indices)


def make_loader(dataset, batch_size, device, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )


def build_default_tasks(num_classes=100, base_classes=30, session=5):
    if num_classes < base_classes:
        raise ValueError('num_classes must be >= base_classes.')
    if session < 1:
        raise ValueError('session must be >= 1.')

    incremental_classes = list(range(base_classes, num_classes))
    if not incremental_classes:
        return []
    if session > len(incremental_classes):
        raise ValueError(
            f'session={session} creates more tasks than incremental classes '
            f'({len(incremental_classes)}).'
        )

    base_size, remainder = divmod(len(incremental_classes), session)
    tasks = []
    cursor = 0
    for task_id in range(session):
        task_size = base_size + (1 if task_id < remainder else 0)
        task_classes = incremental_classes[cursor:cursor + task_size]
        if task_classes:
            tasks.append(task_classes)
        cursor += task_size

    return tasks


def load_backbone(args, device):
    checkpoint = torch.load(args.backbone_ckpt, map_location=device)
    state = checkpoint['backbone'] if isinstance(checkpoint, dict) and 'backbone' in checkpoint else checkpoint

    if args.backbone == 'xception':
        backbone = Xception()
        feature_dim = 2048
    elif args.backbone == 'mantis':
        from backbone.mantisV2.model import MantisV2
        backbone = MantisV2(device=str(device), output_token=args.output_token)
        feature_dim = 512 if args.output_token == 'combined' else 256
    elif args.backbone == 'fcca':
        from backbone.FCCA import FCCABaseModel
        wrapper = FCCABaseModel(
            feature_dim=args.feature_dim,
            num_classes=args.base_classes,
            seq_len=args.slice_len,
            in_channels=args.in_channels,
            cls_scale=args.cls_scale,
            sphere_margin=args.sphere_margin,
        )
        backbone = wrapper.backbone
        feature_dim = args.feature_dim
    elif args.backbone == 'resnet18':
        from backbone.resnet import ResNet18
        backbone = ResNet18()
        feature_dim = args.feature_dim
    else:
        raise ValueError(f'Unknown backbone: {args.backbone}')

    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        log_message(None, f'Backbone load_state_dict strict=False: missing={missing}, unexpected={unexpected}')

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone, feature_dim


def parse_args():
    version=10
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",default=version)
    parser.add_argument('--data-dir', default='/workspace/sunjie/work/data/')

    parser.add_argument('--backbone-ckpt', default=f'best.pth')
    parser.add_argument('--save-dir', default=f'NR-CKA/')
    parser.add_argument('--log-dir', default=f'NR-CKA/')

    parser.add_argument('--dataset', default='jht', choices=['adsb', 'jht', 'tx'])
    parser.add_argument('--backbone', default='xception', choices=['xception', 'mantis', 'fcca', 'resnet18'])
    parser.add_argument('--output-token', default='cls')
    parser.add_argument('--feature-dim', type=int, default=2048)
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--cls-scale', type=float, default=15.0)
    parser.add_argument('--sphere-margin', type=int, default=2)

    parser.add_argument('--slice-len', type=int, default=1000)
    parser.add_argument('--step', type=int, default=800)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)

    parser.add_argument('--num-classes', type=int, default=30)
    parser.add_argument('--base-classes', type=int, default=10)
    parser.add_argument('--session', type=int, nargs='+', default=[5,10,20])

    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda-recon', type=float, default=0.0)
    parser.add_argument('--lambda-recon-margin', type=float, default=0.0)
    parser.add_argument('--lambda-classkey-neg', type=float, default=0.0)
    parser.add_argument('--lambda-class-local', type=float, default=0.0)
    parser.add_argument('--lambda-stage1-margin', type=float, default=0.0)
    parser.add_argument('--lambda-classkey-sep', type=float, default=0.0)
    parser.add_argument('--route-margin', type=float, default=0.2)
    parser.add_argument('--top-k-tasks', type=int, default=10)
    parser.add_argument('--class-route-tau', type=float, default=0.07)
    parser.add_argument('--adapter-alpha', type=float, default=1.0)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--route-rank', type=int, default=8)
    parser.add_argument('--key-prototypes', type=int, default=3)
    parser.add_argument('--logit-calibration', default='zscore', choices=['none', 'center', 'zscore', 'l2'])

    parser.add_argument('--normalize-backbone', action='store_true')
    parser.add_argument('--seed', type=int, nargs='+', default=[41])
    parser.add_argument('--device', default='cuda')

    return parser.parse_args()


def format_route_metrics(prefix, metrics):
    return (
        f"{prefix}_route_acc={metrics['route_vote_acc']:.4f} "
        f"{prefix}_avg_route_acc={metrics['avg_task_vote_acc']:.4f} "
        f"{prefix}_last_route_acc={metrics['last_task_vote_acc']:.4f}"
    )


def format_route_task_metrics(prefix, metrics):
    items = []
    for task_id in sorted(metrics['task_metrics'].keys()):
        items.append(f"{prefix}_task{task_id:02d}_route_acc={metrics['task_metrics'][task_id]['vote_acc']:.4f}")
    return " ".join(items)


def format_class_metrics(prefix, metrics):
    return (
        f"{prefix}_class_acc={metrics['class_vote_acc']:.4f} "
        f"{prefix}_avg_task_class_acc={metrics['avg_task_class_vote_acc']:.4f} "
        f"{prefix}_last_task_class_acc={metrics['last_task_class_vote_acc']:.4f}"
    )


def format_class_task_metrics(prefix, metrics):
    items = []
    for task_id in sorted(metrics['task_metrics'].keys()):
        items.append(f"{prefix}_task{task_id:02d}_class_acc={metrics['task_metrics'][task_id]['vote_acc']:.4f}")
    return " ".join(items)


def format_class_detail_metrics(prefix, metrics):
    items = []
    for class_id in sorted(metrics['class_metrics'].keys()):
        items.append(f"{prefix}_class{class_id:02d}_acc={metrics['class_metrics'][class_id]['vote_acc']:.4f}")
    return " ".join(items)


def write_result_log(run_dir, history):
    path = os.path.join(run_dir, 'result.log')
    cond_order = ['clean', 'snr5', 'snr10', 'snr15', 'snr20']
    lines = []
    for row in history:
        session_id = int(row['session'])
        test_metrics = row.get('test', {})
        for cond_name in cond_order:
            cond_test = test_metrics.get(cond_name, {})
            class_metrics = cond_test.get('class', {})
            lines.append(
                f"session={session_id:02d} {cond_name} "
                f"avg_task_class_acc={class_metrics.get('avg_task_class_vote_acc', 0.0):.4f} "
                f"last_task_class_acc={class_metrics.get('last_task_class_vote_acc', 0.0):.4f}"
            )

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return path


def run_one_session(args, logger, run_dir):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log_message(logger, f'Device: {device}')

    train_set, val_set, test_set = set_dataset(args, logger)
    tasks = build_default_tasks(
        num_classes=args.num_classes,
        base_classes=args.base_classes,
        session=args.session,
    )
    log_message(logger, f'Tasks: {tasks}')

    backbone, feature_dim = load_backbone(args, device)
    log_message(logger, f'Loaded backbone={args.backbone}, feature_dim={feature_dim}')

    model = KVRouteXception(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=args.num_classes,
        rank=args.rank,
        top_k_tasks=args.top_k_tasks,
        adapter_alpha=args.adapter_alpha,
        route_rank=args.route_rank,
        class_route_tau=args.class_route_tau,
        normalize_backbone=args.normalize_backbone,
        key_prototypes=args.key_prototypes,
    ).to(device)
    model.freeze_backbone()

    seen_classes = []
    seen_task_ids = []
    history = []

    for task_id, class_ids in enumerate(tasks):
        model.add_task(task_id, class_ids)
        model.freeze_all_tasks()
        model.unfreeze_route_task(task_id)

        seen_classes.extend(class_ids)
        seen_task_ids.append(task_id)

        train_subset = select_classes(train_set, class_ids)
        test_seen_subset = select_classes(test_set, seen_classes)

        init_loader = make_loader(train_subset, args.batch_size, device, shuffle=False)
        train_loader = make_loader(train_subset, args.batch_size, device, shuffle=True)
        test_loader = make_loader(test_seen_subset, args.batch_size, device, shuffle=False)

        log_message(
            logger,
            f'session={task_id:02d} task={class_ids} seen={len(seen_classes)} '
            f'train_batches={len(train_loader)} test_batches={len(test_loader)}',
        )

        initialize_task_keys(
            model=model,
            loader=init_loader,
            task_id=task_id,
            class_ids=class_ids,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            model.route_trainable_task_parameters(task_id),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        train_logs = train_one_task_kvroute(
            model=model,
            loader=train_loader,
            task_id=task_id,
            seen_task_ids=seen_task_ids,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            lambda_class_local=args.lambda_class_local,
            lambda_stage1_margin=args.lambda_stage1_margin,
            lambda_classkey_sep=args.lambda_classkey_sep,
            top_k_tasks=args.top_k_tasks,
            logger=logger,
        )
        test_conditions = [
            ('clean', None),
            ('snr5', 5),
            ('snr10', 10),
            ('snr15', 15),
            ('snr20', 20),
        ]
        test_metrics = {}
        for cond_name, cond_snr in test_conditions:
            test_metrics[cond_name] = evaluate_stage1_route_class_metrics(
                model=model,
                loader=test_loader,
                seen_task_ids=seen_task_ids,
                device=device,
                top_k_tasks=args.top_k_tasks,
                logit_calibration=args.logit_calibration,
                noise_snr_db=cond_snr,
            )

        row = {
            'session': task_id,
            'task_classes': class_ids,
            'seen_classes': list(seen_classes),
            'train_logs': train_logs,
            'test': test_metrics,
        }
        history.append(row)
        write_result_log(run_dir, history)

        log_message(
            logger,
            f"session={task_id:02d} task={class_ids} seen={len(seen_classes)}"
        )
        for cond_name, _ in test_conditions:
            cond_metrics = test_metrics[cond_name]
            log_message(logger, format_route_metrics(f'{cond_name}_stage1', cond_metrics['route']))
            log_message(logger, format_route_task_metrics(f'{cond_name}_stage1', cond_metrics['route']))
            log_message(logger, format_class_metrics(cond_name, cond_metrics['class']))
            log_message(logger, format_class_task_metrics(cond_name, cond_metrics['class']))
            log_message(logger, format_class_detail_metrics(cond_name, cond_metrics['class']))

    if history:
        final_test = history[-1]['test']
        for cond_name in ['clean', 'snr5', 'snr10', 'snr15', 'snr20']:
            cond_test = final_test[cond_name]
            log_message(
                logger,
                f"final_test_stage1_{cond_name} "
                f"route_acc={cond_test['route']['route_vote_acc']:.4f} "
                f"avg_route_acc={cond_test['route']['avg_task_vote_acc']:.4f} "
                f"last_route_acc={cond_test['route']['last_task_vote_acc']:.4f} "
                f"class_acc={cond_test['class']['class_vote_acc']:.4f} "
                f"avg_task_class_acc={cond_test['class']['avg_task_class_vote_acc']:.4f} "
                f"last_task_class_acc={cond_test['class']['last_task_class_vote_acc']:.4f}",
            )

    save_run_dir = os.path.join(
        args.save_dir,
        args.backbone,
        f'session_{args.session}',
        args.dataset,
        os.path.basename(run_dir),
    )
    os.makedirs(save_run_dir, exist_ok=True)
    save_path = os.path.join(save_run_dir, 'kvroute_incremental.pt')
    torch.save(
        {
            'model': model.state_dict(),
            'history': history,
            'tasks': tasks,
            'args': vars(args),
        },
        save_path,
    )
    log_message(logger, f'Saved KVRoute checkpoint to {save_path}')

    result_log_path = write_result_log(run_dir, history)
    log_message(logger, f'Saved compact result log to {result_log_path}')

    final_test = history[-1]['test'] if history else None
    final_score = 0.0 if final_test is None else final_test['clean']['class']['avg_task_class_vote_acc']
    return {
        'history': history,
        'tasks': tasks,
        'save_path': save_path,
        'result_log_path': result_log_path,
        'final_test': final_test,
        'final_score': final_score,
    }



def main():
    args = parse_args()
    session_values = args.session if isinstance(args.session, list) else [args.session]
    seed_values = args.seed if isinstance(args.seed, list) else [args.seed]

    for session in session_values:
        session = int(session)
        base_runtime = datetime.now().strftime('%Y%m%d_%H%M%S')
        for seed in seed_values:
            seed = int(seed)

            run_args = argparse.Namespace(**vars(args))
            run_args.session = session
            run_args.seed = seed

            set_seed(run_args.seed)
            runtime = f"{base_runtime}_seed{seed}"
            logger, run_dir = set_logger(runtime, run_args.log_dir, run_args.backbone, run_args)
            log_message(
                logger,
                f"Running session split={run_args.session} seed={run_args.seed}",
            )
            outcome = run_one_session(run_args, logger, run_dir)
            final_test = outcome['final_test'] or {}
            clean_metrics = final_test.get('clean', {})
            clean_class_metrics = clean_metrics.get('class', {})
            clean_route_metrics = clean_metrics.get('route', {})
            result = {
                'session': session,
                'seed': seed,
                'final_score': outcome['final_score'],
                'final_class_acc': clean_class_metrics.get('class_vote_acc', 0.0),
                'final_avg_task_class_acc': clean_class_metrics.get('avg_task_class_vote_acc', 0.0),
                'final_last_task_class_acc': clean_class_metrics.get('last_task_class_vote_acc', 0.0),
                'final_route_acc': clean_route_metrics.get('route_vote_acc', 0.0),
                'final_avg_route_acc': clean_route_metrics.get('avg_task_vote_acc', 0.0),
                'final_last_route_acc': clean_route_metrics.get('last_task_vote_acc', 0.0),
                'save_path': outcome['save_path'],
                'final_clean_test': clean_metrics,
                'final_noisy_tests': {k: v for k, v in final_test.items() if k != 'clean'},
            }
            result_log_path = outcome.get('result_log_path')
            log_message(
                logger,
                f"run_result seed={seed} "
                f"score={result['final_score']:.4f} "
                f"class_acc={result['final_class_acc']:.4f} "
                f"avg_route_acc={result['final_avg_route_acc']:.4f} "
                f"result_log={result_log_path}",
            )


if __name__ == '__main__':
    main()
