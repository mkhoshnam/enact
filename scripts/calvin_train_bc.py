import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import zarr
from torchvision.models import ResNet18_Weights, resnet18


OUT_BASE = Path("/path/to/enact_calvin_outputs")
DATASET_DIR = OUT_BASE / "calvin"
ZARR_PATH = DATASET_DIR / "training_dataset_future_bc.zarr"
OUT_DIR = OUT_BASE / "calvin_bc"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4
BACKBONE_LR = 2e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True
VAL_SEGMENT_RATIO = 0.10

HIDDEN_DIM = 384
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.10
FF_MULT = 4
GRIPPER_LOSS_WEIGHT = 0.25
PROGRESS_LOSS_WEIGHT = 0.35
LATE_STEP_WEIGHT_END = 2.0
SAVE_EVERY = 5
USE_PRETRAINED_BACKBONE = True

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def build_segment_split(z, val_ratio=0.10, seed=42):
    seg_ids = np.asarray(z["segment_id"][:], dtype=np.int64)
    unique_seg_ids = np.unique(seg_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_seg_ids)
    n_val = max(1, int(len(unique_seg_ids) * val_ratio))
    val_seg = set(unique_seg_ids[:n_val].tolist())
    train_seg = set(unique_seg_ids[n_val:].tolist())
    all_idx = np.arange(len(seg_ids), dtype=np.int64)
    train_idx = all_idx[np.isin(seg_ids, list(train_seg))]
    val_idx = all_idx[np.isin(seg_ids, list(val_seg))]
    return train_idx, val_idx


class CalvinFutureDataset(Dataset):
    def __init__(self, z, indices):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.obs_static = z["obs_static_frames"]
        self.obs_gripper = z["obs_gripper_frames"]
        self.future_static = z["future_static_frames"]
        self.arm_action = z["action_chunk"]
        self.gripper = z["gripper_chunk"]
        self.task_id = z["task_id"]
        self.goal_type = z["goal_type"]
        self.arm_mean = np.asarray(z["stats/arm_action_mean"][:], dtype=np.float32)
        self.arm_std = np.asarray(z["stats/arm_action_std"][:], dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        arm = np.asarray(self.arm_action[i], dtype=np.float32)
        arm_n = (arm - self.arm_mean[None, :]) / self.arm_std[None, :]
        return {
            "obs_static": torch.from_numpy(np.asarray(self.obs_static[i], dtype=np.uint8)),
            "obs_gripper": torch.from_numpy(np.asarray(self.obs_gripper[i], dtype=np.uint8)),
            "future_static": torch.from_numpy(np.asarray(self.future_static[i], dtype=np.uint8)),
            "task_id": torch.tensor(int(self.task_id[i]), dtype=torch.long),
            "goal_type": torch.tensor(int(self.goal_type[i]), dtype=torch.long),
            "arm_action": torch.from_numpy(arm_n.astype(np.float32)),
            "gripper": torch.from_numpy(np.asarray(self.gripper[i], dtype=np.float32)),
        }


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        return self.feature_extractor(x).flatten(1)


class FutureBC(nn.Module):
    def __init__(self, arm_dim, chunk_horizon, obs_horizon, future_horizon, num_tasks, hidden_dim=384, num_layers=4, num_heads=8, dropout=0.10, ff_mult=4, pretrained_backbone=True):
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.obs_horizon = int(obs_horizon)
        self.future_horizon = int(future_horizon)
        self.num_tasks = int(num_tasks)

        self.image_encoder = ResNet18Encoder(pretrained=pretrained_backbone)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, hidden_dim)
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.goal_emb = nn.Embedding(4, hidden_dim)
        self.action_queries = nn.Parameter(torch.zeros(1, self.chunk_horizon, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        total_tokens = self.chunk_horizon + 2 + self.obs_horizon + self.obs_horizon + self.future_horizon
        self.pos_emb = nn.Parameter(torch.zeros(1, total_tokens, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.modality_emb = nn.Embedding(6, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.arm_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, self.arm_dim))
        self.gripper_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def encode_frames(self, frames_u8):
        b, t, h, w, c = frames_u8.shape
        x = frames_u8.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        x = x.view(b * t, c, h, w)
        feat = self.image_proj(self.image_encoder(x))
        return feat.view(b, t, -1)

    def encode_tokens(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        bsz = obs_static.shape[0]
        tok_actions = self.action_queries.expand(bsz, -1, -1)
        tok_task = self.task_emb(task_id).unsqueeze(1)
        tok_goal = self.goal_emb(goal_type).unsqueeze(1)
        tok_obs_static = self.encode_frames(obs_static)
        tok_obs_gripper = self.encode_frames(obs_gripper)
        tok_future = self.encode_frames(future_static)
        x = torch.cat([tok_actions, tok_task, tok_goal, tok_obs_static, tok_obs_gripper, tok_future], dim=1)
        token_types = ([0] * self.chunk_horizon + [1] + [2] + [3] * self.obs_horizon + [4] * self.obs_horizon + [5] * self.future_horizon)
        token_types = torch.tensor(token_types, dtype=torch.long, device=x.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.modality_emb(token_types) + self.pos_emb[:, :x.shape[1], :]
        x = self.transformer(self.input_norm(x))
        return self.output_norm(x[:, :self.chunk_horizon])

    def forward(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        z = self.encode_tokens(obs_static, obs_gripper, future_static, task_id, goal_type)
        return self.arm_head(z), self.gripper_head(z).squeeze(-1)

    def forward_with_latent(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        z = self.encode_tokens(obs_static, obs_gripper, future_static, task_id, goal_type)
        return z, self.arm_head(z), self.gripper_head(z).squeeze(-1)


def arm_loss_fn(pred, target):
    elem = F.smooth_l1_loss(pred, target, reduction="none")
    weights = torch.linspace(1.0, LATE_STEP_WEIGHT_END, steps=pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, pred.shape[1], 1)
    action_loss = (elem * weights).mean()
    progress_loss = F.smooth_l1_loss(pred.sum(dim=1), target.sum(dim=1), reduction="mean")
    return action_loss, progress_loss


def run_epoch(model, loader, optimizer, grip_criterion, train=True):
    model.train(train)
    totals = {"loss": 0.0, "arm": 0.0, "progress": 0.0, "grip": 0.0, "grip_acc": 0.0, "count": 0}
    for batch in loader:
        obs_static = batch["obs_static"].to(DEVICE, non_blocking=True)
        obs_gripper = batch["obs_gripper"].to(DEVICE, non_blocking=True)
        future_static = batch["future_static"].to(DEVICE, non_blocking=True)
        task_id = batch["task_id"].to(DEVICE, non_blocking=True)
        goal_type = batch["goal_type"].to(DEVICE, non_blocking=True)
        arm_action = batch["arm_action"].to(DEVICE, non_blocking=True)
        gripper = batch["gripper"].to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            arm_pred, grip_logits = model(obs_static, obs_gripper, future_static, task_id, goal_type)
            arm_loss, progress_loss = arm_loss_fn(arm_pred, arm_action)
            grip_loss = grip_criterion(grip_logits, gripper)
            loss = arm_loss + PROGRESS_LOSS_WEIGHT * progress_loss + GRIPPER_LOSS_WEIGHT * grip_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.no_grad():
            grip_acc = ((torch.sigmoid(grip_logits) > 0.5) == (gripper > 0.5)).float().mean()
        bs = int(obs_static.shape[0])
        totals["loss"] += float(loss.item()) * bs
        totals["arm"] += float(arm_loss.item()) * bs
        totals["progress"] += float(progress_loss.item()) * bs
        totals["grip"] += float(grip_loss.item()) * bs
        totals["grip_acc"] += float(grip_acc.item()) * bs
        totals["count"] += bs
    n = max(totals["count"], 1)
    return {k: totals[k] / n for k in ["loss", "arm", "progress", "grip", "grip_acc"]}


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    if not ZARR_PATH.exists():
        raise FileNotFoundError("Dataset not found: {}".format(ZARR_PATH))

    z = zarr.open(str(ZARR_PATH), mode="r")
    tasks = list(z.attrs.get("tasks", []))
    if len(tasks) == 0:
        raise RuntimeError("Dataset does not contain task list")

    train_idx, val_idx = build_segment_split(z, VAL_SEGMENT_RATIO, SEED)
    train_ds = CalvinFutureDataset(z, train_idx)
    val_ds = CalvinFutureDataset(z, val_idx)
    sample = train_ds[0]

    arm_dim = int(sample["arm_action"].shape[-1])
    chunk_horizon = int(sample["arm_action"].shape[0])
    obs_horizon = int(sample["obs_static"].shape[0])
    future_horizon = int(sample["future_static"].shape[0])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = FutureBC(
        arm_dim=arm_dim,
        chunk_horizon=chunk_horizon,
        obs_horizon=obs_horizon,
        future_horizon=future_horizon,
        num_tasks=len(tasks),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        ff_mult=FF_MULT,
        pretrained_backbone=USE_PRETRAINED_BACKBONE,
    ).to(DEVICE)

    backbone_params = list(model.image_encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": LR},
        {"params": backbone_params, "lr": BACKBONE_LR},
    ], weight_decay=WEIGHT_DECAY)
    grip_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    cfg = {
        "tasks": tasks,
        "zarr_path": str(ZARR_PATH),
        "out_dir": str(OUT_DIR),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "backbone_lr": BACKBONE_LR,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "ff_mult": FF_MULT,
        "chunk_horizon": chunk_horizon,
        "arm_dim": arm_dim,
        "obs_horizon": obs_horizon,
        "future_horizon": future_horizon,
        "num_tasks": len(tasks),
    }
    with open(OUT_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_log = run_epoch(model, train_loader, optimizer, grip_criterion, train=True)
        val_log = run_epoch(model, val_loader, optimizer, grip_criterion, train=False)
        row = {"epoch": epoch, "train": train_log, "val": val_log}
        history.append(row)
        print("epoch {:03d} train {:.5f} val {:.5f} grip {:.3f}".format(epoch, train_log["loss"], val_log["loss"], val_log["grip_acc"]))

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": min(best_val, val_log["loss"]),
            "config": cfg,
            "arm_action_mean": train_ds.arm_mean,
            "arm_action_std": train_ds.arm_std,
        }
        if val_log["loss"] < best_val:
            best_val = val_log["loss"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, OUT_DIR / "bc_actor_best.pt")
        if epoch % SAVE_EVERY == 0:
            torch.save(ckpt, OUT_DIR / "bc_actor_epoch_{:03d}.pt".format(epoch))
        with open(OUT_DIR / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("done", OUT_DIR / "bc_actor_best.pt")


if __name__ == "__main__":
    main()
