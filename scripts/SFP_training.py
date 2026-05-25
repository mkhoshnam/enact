import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


REPO_ROOT = Path(__file__).resolve().parents[1]


TASKS = [
    "open_drawer",
    "close_drawer",
    "push_into_drawer",
    "turn_on_led",
    "turn_off_led",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "move_slider_left",
]


def env_path(name, default=None):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


OUT_BASE = env_path("ENACT_CALVIN_OUT_BASE", REPO_ROOT / "outputs")
DATASET_DIR = OUT_BASE / "calvin"
ZARR_PATH = DATASET_DIR / "training_dataset_future_bc.zarr"
OUT_DIR = OUT_BASE / "calvin_sfp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 300
LR = 5e-5
BACKBONE_LR = 0.0
WEIGHT_DECAY = 1e-6
NUM_WORKERS = 4
PIN_MEMORY = True
VAL_SEGMENT_RATIO = 0.10
SAVE_EVERY = 10

IMAGE_EMBED_DIM = 64
TASK_EMBED_DIM = 32
GOAL_EMBED_DIM = 16
DOWN_DIMS = (256, 512, 1024)
KERNEL_SIZE = 5
N_GROUPS = 8
SIGMA0 = 0.4
FLOW_GAIN = 10.0
EMA_DECAY = 0.75
GRIPPER_LOSS_WEIGHT = 1.0

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


def compute_minmax(arr, chunk_size=65536):
    first = np.asarray(arr[0], dtype=np.float32)
    amin = np.full(first.shape[-1], np.inf, dtype=np.float64)
    amax = np.full(first.shape[-1], -np.inf, dtype=np.float64)
    for start in range(0, int(arr.shape[0]), int(chunk_size)):
        x = np.asarray(arr[start:start + chunk_size], dtype=np.float32).reshape(-1, first.shape[-1])
        amin = np.minimum(amin, x.min(axis=0))
        amax = np.maximum(amax, x.max(axis=0))
    return amin.astype(np.float32), amax.astype(np.float32)


def normalize_minmax(data, amin, amax):
    denom = np.maximum(amax - amin, 1e-6)
    return ((data - amin) / denom * 2.0 - 1.0).astype(np.float32)


def unnormalize_minmax(data, amin, amax):
    denom = np.maximum(amax - amin, 1e-6)
    return ((data + 1.0) * 0.5 * denom + amin).astype(np.float32)


class CalvinSFPDataset(Dataset):
    def __init__(self, z, indices, arm_min, arm_max):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.obs_static = z["obs_static_frames"]
        self.obs_gripper = z["obs_gripper_frames"]
        self.future_static = z["future_static_frames"]
        self.arm_action = z["action_chunk"]
        self.gripper = z["gripper_chunk"]
        self.task_id = z["task_id"]
        self.goal_type = z["goal_type"]
        self.arm_min = np.asarray(arm_min, dtype=np.float32)
        self.arm_max = np.asarray(arm_max, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = int(self.indices[item])
        arm = np.asarray(self.arm_action[i], dtype=np.float32)
        return {
            "obs_static": torch.from_numpy(np.asarray(self.obs_static[i], dtype=np.uint8)),
            "obs_gripper": torch.from_numpy(np.asarray(self.obs_gripper[i], dtype=np.uint8)),
            "future_static": torch.from_numpy(np.asarray(self.future_static[i], dtype=np.uint8)),
            "task_id": torch.tensor(int(self.task_id[i]), dtype=torch.long),
            "goal_type": torch.tensor(int(self.goal_type[i]), dtype=torch.long),
            "arm_action": torch.from_numpy(normalize_minmax(arm, self.arm_min[None, :], self.arm_max[None, :])),
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


class SFPConditionEncoder(nn.Module):
    def __init__(self, obs_horizon, future_horizon, num_tasks, image_embed_dim=64, task_embed_dim=32, goal_embed_dim=16, pretrained_backbone=True):
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        self.future_horizon = int(future_horizon)
        self.num_tasks = int(num_tasks)
        self.image_embed_dim = int(image_embed_dim)
        self.task_embed_dim = int(task_embed_dim)
        self.goal_embed_dim = int(goal_embed_dim)

        self.image_encoder = ResNet18Encoder(pretrained=pretrained_backbone)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, self.image_embed_dim)
        self.task_emb = nn.Embedding(self.num_tasks, self.task_embed_dim)
        self.goal_emb = nn.Embedding(4, self.goal_embed_dim)

        self.obs_dim = self.image_embed_dim * 2
        self.future_dim = self.image_embed_dim
        self.out_dim = self.obs_dim * self.obs_horizon + self.future_dim + self.task_embed_dim + self.goal_embed_dim

    def encode_frames(self, frames_u8):
        b, t, h, w, c = frames_u8.shape
        x = frames_u8.float() / 255.0
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        x = x.view(b * t, c, h, w)
        feat = self.image_proj(self.image_encoder(x))
        return feat.view(b, t, -1)

    def forward(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        static_tokens = self.encode_frames(obs_static)
        gripper_tokens = self.encode_frames(obs_gripper)
        obs_tokens = torch.cat([static_tokens, gripper_tokens], dim=-1)
        future_tokens = self.encode_frames(future_static).mean(dim=1)
        task = self.task_emb(task_id)
        goal = self.goal_emb(goal_type)
        return torch.cat([obs_tokens.flatten(start_dim=1), future_tokens, task, goal], dim=-1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = int(dim)
        self.scale = float(scale)

    def forward(self, x):
        x = x * self.scale
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConvUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.out_channels = int(out_channels)
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=(256, 512, 1024), kernel_size=5, n_groups=8, sin_embedding_scale=1.0):
        super().__init__()
        all_dims = [int(input_dim)] + list(down_dims)
        start_dim = int(down_dims[0])
        dsed = int(diffusion_step_embed_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale=sin_embedding_scale),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + int(global_cond_dim)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConvDownsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConvUpsample1d(dim_in) if not is_last else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: Tensor, timestep, global_cond=None):
        x = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=x.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.float().expand(x.shape[0])
        cond = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            cond = torch.cat([cond, global_cond], dim=-1)
        h = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            h.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for res1, res2, upsample in self.up_modules:
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                min_t = min(x.shape[-1], skip.shape[-1])
                x = x[..., :min_t]
                skip = skip[..., :min_t]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)
        return self.final_conv(x).moveaxis(-1, -2)


class GripperHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Mish(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EMA(object):
    def __init__(self, model, decay=0.75):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for shadow_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(model_b)


def linearly_interpolate_trajectory(xi, t):
    batch_size, horizon, dim = xi.shape
    scaled_t = t * (horizon - 1)
    lo = scaled_t.floor().long().clamp(0, horizon - 2)
    hi = (lo + 1).clamp(0, horizon - 1)
    frac = (scaled_t - lo.float()).unsqueeze(-1)
    batch_idx = torch.arange(batch_size, device=xi.device)
    xi_lo = xi[batch_idx, lo]
    xi_hi = xi[batch_idx, hi]
    xi_t = xi_lo + frac * (xi_hi - xi_lo)
    dxi_dt = (xi_hi - xi_lo) * (horizon - 1)
    return xi_t, dxi_dt


def sample_cfm_inputs_and_targets(xi_t, dxi_dt, t, sigma0, gain):
    noise_scale = float(sigma0) * torch.exp(-float(gain) * t).unsqueeze(-1)
    noise = noise_scale * torch.randn_like(xi_t)
    a = xi_t + noise
    v = -float(gain) * noise + dxi_dt
    return a.unsqueeze(1), v.unsqueeze(1)


def make_models(sample, tasks):
    arm_dim = int(sample["arm_action"].shape[-1])
    action_horizon = int(sample["arm_action"].shape[0])
    obs_horizon = int(sample["obs_static"].shape[0])
    future_horizon = int(sample["future_static"].shape[0])
    condition_encoder = SFPConditionEncoder(
        obs_horizon=obs_horizon,
        future_horizon=future_horizon,
        num_tasks=len(tasks),
        image_embed_dim=IMAGE_EMBED_DIM,
        task_embed_dim=TASK_EMBED_DIM,
        goal_embed_dim=GOAL_EMBED_DIM,
        pretrained_backbone=True,
    ).to(DEVICE)
    velocity_model = ConditionalUnet1D(
        input_dim=arm_dim,
        global_cond_dim=condition_encoder.out_dim,
        down_dims=DOWN_DIMS,
        kernel_size=KERNEL_SIZE,
        n_groups=N_GROUPS,
        sin_embedding_scale=1.0,
    ).to(DEVICE)
    gripper_head = GripperHead(condition_encoder.out_dim + 1).to(DEVICE)
    return condition_encoder, velocity_model, gripper_head, arm_dim, action_horizon, obs_horizon, future_horizon


def run_epoch(condition_encoder, velocity_model, gripper_head, loader, optimizer, train=True):
    condition_encoder.train(train)
    velocity_model.train(train)
    gripper_head.train(train)
    totals = {"loss": 0.0, "arm": 0.0, "grip": 0.0, "grip_acc": 0.0, "count": 0}
    for batch in loader:
        obs_static = batch["obs_static"].to(DEVICE, non_blocking=True)
        obs_gripper = batch["obs_gripper"].to(DEVICE, non_blocking=True)
        future_static = batch["future_static"].to(DEVICE, non_blocking=True)
        task_id = batch["task_id"].to(DEVICE, non_blocking=True)
        goal_type = batch["goal_type"].to(DEVICE, non_blocking=True)
        arm_action = batch["arm_action"].to(DEVICE, non_blocking=True)
        gripper = batch["gripper"].to(DEVICE, non_blocking=True)
        batch_size = int(arm_action.shape[0])

        with torch.set_grad_enabled(train):
            gcond = condition_encoder(obs_static, obs_gripper, future_static, task_id, goal_type)
            t = torch.rand(batch_size, device=DEVICE)
            xi_t, dxi_dt = linearly_interpolate_trajectory(arm_action, t)
            a, v = sample_cfm_inputs_and_targets(xi_t, dxi_dt, t, SIGMA0, FLOW_GAIN)
            v_hat = velocity_model(a, t, gcond)
            arm_loss = F.mse_loss(v_hat, v)

            idx = (t * (arm_action.shape[1] - 1)).round().long().clamp(0, arm_action.shape[1] - 1)
            grip_target = gripper[torch.arange(batch_size, device=DEVICE), idx]
            grip_logits = gripper_head(torch.cat([gcond, t.unsqueeze(-1)], dim=-1))
            pos = grip_target.sum().clamp(min=1.0)
            neg = (grip_target.numel() - grip_target.sum()).clamp(min=1.0)
            grip_loss = F.binary_cross_entropy_with_logits(grip_logits, grip_target, pos_weight=(neg / pos).detach())
            loss = arm_loss + GRIPPER_LOSS_WEIGHT * grip_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(condition_encoder.parameters()) + list(velocity_model.parameters()) + list(gripper_head.parameters()),
                    max_norm=5.0,
                )
                optimizer.step()

        with torch.no_grad():
            grip_acc = ((torch.sigmoid(grip_logits) > 0.5) == (grip_target > 0.5)).float().mean()
        totals["loss"] += float(loss.item()) * batch_size
        totals["arm"] += float(arm_loss.item()) * batch_size
        totals["grip"] += float(grip_loss.item()) * batch_size
        totals["grip_acc"] += float(grip_acc.item()) * batch_size
        totals["count"] += batch_size
    n = max(totals["count"], 1)
    return {k: totals[k] / n for k in ["loss", "arm", "grip", "grip_acc"]}


def save_checkpoint(path, condition_encoder, velocity_model, gripper_head, optimizer, cfg, arm_min, arm_max, epoch, best_val):
    ensure_dir(path.parent)
    torch.save(
        {
            "condition_encoder_state_dict": condition_encoder.state_dict(),
            "velocity_model_state_dict": velocity_model.state_dict(),
            "gripper_head_state_dict": gripper_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "best_val": float(best_val),
            "config": cfg,
            "arm_action_min": np.asarray(arm_min, dtype=np.float32),
            "arm_action_max": np.asarray(arm_max, dtype=np.float32),
            "policy_type": "sfp",
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFP for all CALVIN future-conditioned tasks")
    parser.add_argument("--zarr_path", default=str(ZARR_PATH))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--backbone_lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    return parser.parse_args()


def main():
    global SEED, BATCH_SIZE, EPOCHS, LR, BACKBONE_LR, NUM_WORKERS, SAVE_EVERY
    args = parse_args()
    SEED = int(args.seed)
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    LR = float(args.lr)
    BACKBONE_LR = float(args.backbone_lr)
    NUM_WORKERS = int(args.num_workers)
    SAVE_EVERY = int(args.save_every)

    set_seed(SEED)
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(out_dir)
    zarr_path = Path(args.zarr_path).expanduser()
    if not zarr_path.exists():
        raise FileNotFoundError("Dataset not found: {}".format(zarr_path))

    z = zarr.open(str(zarr_path), mode="r")
    tasks = list(z.attrs.get("tasks", TASKS))
    if len(tasks) != 8:
        print("[WARN] expected 8 CALVIN tasks, found {}: {}".format(len(tasks), tasks))
    train_idx, val_idx = build_segment_split(z, VAL_SEGMENT_RATIO, SEED)
    arm_min, arm_max = compute_minmax(z["action_chunk"])
    train_ds = CalvinSFPDataset(z, train_idx, arm_min, arm_max)
    val_ds = CalvinSFPDataset(z, val_idx, arm_min, arm_max)
    sample = train_ds[0]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    condition_encoder, velocity_model, gripper_head, arm_dim, action_horizon, obs_horizon, future_horizon = make_models(sample, tasks)

    backbone_params = list(condition_encoder.image_encoder.parameters())
    for p in backbone_params:
        p.requires_grad = BACKBONE_LR > 0.0
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in list(condition_encoder.parameters()) + list(velocity_model.parameters()) + list(gripper_head.parameters()) if id(p) not in backbone_ids]
    param_groups = [{"params": other_params, "lr": LR}]
    if BACKBONE_LR > 0.0:
        param_groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    ema_velocity = EMA(velocity_model, EMA_DECAY)

    cfg = {
        "policy_type": "sfp",
        "tasks": tasks,
        "zarr_path": str(zarr_path),
        "out_dir": str(out_dir),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "backbone_lr": BACKBONE_LR,
        "sigma0": SIGMA0,
        "flow_gain": FLOW_GAIN,
        "ema_decay": EMA_DECAY,
        "image_embed_dim": IMAGE_EMBED_DIM,
        "task_embed_dim": TASK_EMBED_DIM,
        "goal_embed_dim": GOAL_EMBED_DIM,
        "condition_dim": condition_encoder.out_dim,
        "arm_dim": arm_dim,
        "action_horizon": action_horizon,
        "obs_horizon": obs_horizon,
        "future_horizon": future_horizon,
        "num_tasks": len(tasks),
        "down_dims": list(DOWN_DIMS),
        "kernel_size": KERNEL_SIZE,
        "n_groups": N_GROUPS,
    }
    with open(out_dir / "sfp_train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_log = run_epoch(condition_encoder, velocity_model, gripper_head, train_loader, optimizer, train=True)
        ema_velocity.update(velocity_model)
        val_log = run_epoch(condition_encoder, ema_velocity.shadow, gripper_head, val_loader, optimizer, train=False)
        row = {"epoch": epoch, "train": train_log, "val": val_log}
        history.append(row)
        print("epoch {:04d} train {:.5f} val {:.5f} grip {:.3f}".format(epoch, train_log["loss"], val_log["loss"], val_log["grip_acc"]))

        if val_log["loss"] < best_val:
            best_val = val_log["loss"]
            save_checkpoint(out_dir / "sfp_policy_best.pt", condition_encoder, ema_velocity.shadow, gripper_head, optimizer, cfg, arm_min, arm_max, epoch, best_val)
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(out_dir / "sfp_policy_epoch_{:04d}.pt".format(epoch), condition_encoder, ema_velocity.shadow, gripper_head, optimizer, cfg, arm_min, arm_max, epoch, best_val)
        with open(out_dir / "sfp_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(out_dir / "sfp_policy_final.pt", condition_encoder, ema_velocity.shadow, gripper_head, optimizer, cfg, arm_min, arm_max, EPOCHS, best_val)
    print("done", out_dir / "sfp_policy_best.pt")


if __name__ == "__main__":
    main()
