import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import json
import os
import random
import re
import yaml
from copy import deepcopy
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import hydra
import pybullet as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.models import ResNet18_Weights, resnet18


REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default=None):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


SEED = int(os.environ.get("CALVIN_SEED", "42"))

CALVIN_ROOT = env_path("CALVIN_ROOT")
DATA_ROOT = env_path(
    "CALVIN_DATA_ROOT",
    CALVIN_ROOT / "dataset/task_D_D/training" if CALVIN_ROOT is not None else None,
)
OUT_BASE = env_path("ENACT_CALVIN_OUT_BASE", REPO_ROOT / "outputs")
SEGMENTS_JSON = env_path("CALVIN_SEGMENTS_JSON", OUT_BASE / "calvin" / "segments_future_bc.json")
BC_CKPT_PATH = env_path("CALVIN_BC_CKPT_PATH", OUT_BASE / "calvin_bc" / "bc_actor_best.pt")
RESULTS_ROOT = env_path("CALVIN_RESULTS_ROOT", OUT_BASE)
RUN_NAME = os.environ.get(
    "CALVIN_RL_RUN_NAME",
    "multitask_step6_rafc_td3bc_seed{}".format(SEED),
)
RESULTS_DIR = env_path("CALVIN_RL_RESULTS_DIR", RESULTS_ROOT / "rafc_rl_runs" / RUN_NAME)
FINAL_ACTOR_PATH = env_path("CALVIN_RL_FINAL_ACTOR_PATH", RESULTS_DIR / "policy_final.pt")
BEST_ACTOR_PATH = env_path("CALVIN_RL_BEST_ACTOR_PATH", RESULTS_DIR / "policy_best.pt")
HISTORY_JSON = RESULTS_DIR / "history.json"

GENERATED_FUTURE_ROOT = env_path("CALVIN_GENERATED_FUTURE_ROOT", OUT_BASE / "generated_inpainted_calvin_futures")
USE_GENERATED_FUTURES_DURING_RL = os.environ.get("CALVIN_USE_GENERATED_FUTURES_DURING_RL", "0") == "1"
USE_RAFC = os.environ.get("CALVIN_USE_RAFC", "1") != "0"
FUTURE_SHIFTS = (-2, 0, 2)
RAFC_INIT_ALPHA = 0.90
RAFC_CENTER_LOGIT_BIAS = 3.0
RAFC_ALPHA_REG = 0.0
RAFC_ENTROPY_REG = 0.0
RAFC_ALPHA_TARGET = 0.70

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.90
IMAGE_SIZE = None

MAX_EPISODE_STEPS = 150
TOTAL_ENV_STEPS = 140_000
START_TRAIN_AFTER = 2_000
RANDOM_WARMUP_STEPS = 4_000
BUFFER_SIZE = 150_000
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.05
NOISE_CLIP = 0.08
POLICY_DELAY = 2
ACTOR_LR = 5e-5
CRITIC_LR = 5e-5
WEIGHT_DECAY = 1e-5

ARM_ACTION_DIM = 6
DELTA_ARM_LIMIT = 0.10
GRIP_DELTA_LIMIT = 2.0
ACTION_DIM = ARM_ACTION_DIM + 1
EXPL_NOISE_STD = 0.04
EXPL_NOISE_CLIP = 0.08
ACTION_REPEAT = 2

SUCCESS_REWARD = 120.0
STEP_PENALTY = -0.005
TIMEOUT_PENALTY = -25.0
BROKEN_PENALTY = -20.0
INDEX_PROGRESS_SCALE = 3.0
ROBOT_PROGRESS_SCALE = 4.0
SCENE_PROGRESS_SCALE = 2.5
DELTA_L2 = 0.02

EVAL_EVERY_STEPS = 5_000
NUM_EVAL_EPISODES = 12
SHOW_GUI_TRAIN = False
SHOW_GUI_EVAL = False

ALIGN_BACKWARD_SEARCH = 2
ALIGN_FORWARD_SEARCH = 5
ALIGN_MAX_FORWARD_STEP = 2
ALIGN_MAX_BACKWARD_STEP = 0
PHASE_INIT_STEPS = 15

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
_EP_RE = re.compile(r"episode_(\d+)\.npz$")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def patch_yaml_tags_for_omegaconf():
    try:
        yaml.SafeLoader.add_constructor("!tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    except Exception:
        pass


def compose_cfg():
    if DATA_ROOT is None:
        raise RuntimeError("Set CALVIN_DATA_ROOT or CALVIN_ROOT before creating the CALVIN environment")
    conf_path = DATA_ROOT / ".hydra" / "merged_config.yaml"
    if not conf_path.exists():
        raise FileNotFoundError("Could not find merged_config at {}".format(conf_path))
    patch_yaml_tags_for_omegaconf()
    return OmegaConf.load(conf_path)


def make_env(show_gui):
    cfg = compose_cfg()
    try:
        OmegaConf.set_struct(cfg, False)
        cfg.env.show_gui = bool(show_gui)
        cfg.env.use_egl = False
        cfg.env.use_scene_info = True
    except Exception:
        pass
    try:
        p.disconnect()
    except Exception:
        pass
    env = hydra.utils.instantiate(cfg.env)
    tasks_oracle = hydra.utils.instantiate(cfg.tasks)
    return env, tasks_oracle


def oracle_success(tasks_oracle, start_info, curr_info, task_key):
    ti = tasks_oracle.get_task_info(start_info, curr_info)
    if isinstance(ti, dict):
        return bool(ti.get(task_key, False))
    try:
        return task_key in ti
    except Exception:
        return False


def build_episode_file_map(data_root):
    files = sorted(data_root.glob("episode_*.npz"))
    if len(files) == 0:
        raise RuntimeError("No episode_*.npz files found in {}".format(data_root))
    out = {}
    for pth in files:
        m = _EP_RE.search(pth.name)
        if m is not None:
            out[int(m.group(1))] = pth
    return out


def infer_action_key(item):
    if "rel_actions" in item:
        return "rel_actions"
    if "actions" in item:
        return "actions"
    raise KeyError("No action key. Available: {}".format(list(item.keys())))


class EpisodeCache(object):
    def __init__(self, data_root, max_items=2048):
        self.data_root = data_root
        self.max_items = int(max_items)
        self.episode_file_map = build_episode_file_map(data_root)
        self.cache = OrderedDict()
        self.action_key = None

    def get(self, idx):
        idx = int(idx)
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx not in self.episode_file_map:
            raise KeyError("Episode index {} not found".format(idx))
        data = np.load(str(self.episode_file_map[idx]), allow_pickle=True)
        item = {k: data[k] for k in data.files}
        if self.action_key is None:
            self.action_key = infer_action_key(item)
        self.cache[idx] = item
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return self.cache[idx]


def normalize_text(s):
    s = str(s).lower().strip().replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_segments(path):
    if not path.exists():
        raise FileNotFoundError("Missing segments json: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "segments" in obj:
        segments = obj["segments"]
        tasks = obj.get("tasks", None)
        task_to_id = obj.get("task_to_id", None)
    else:
        segments = obj
        tasks = None
        task_to_id = None

    if tasks is None:
        tasks = sorted({
            normalize_text(s["task"]).replace(" ", "_")
            for s in segments
            if "task" in s
        })
    if task_to_id is None:
        task_to_id = {t: i for i, t in enumerate(tasks)}

    return segments, tasks, task_to_id


def split_segments(segments, train_split, seed):
    ids = np.arange(len(segments))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_split)))
    n_train = min(n_train, len(ids) - 1) if len(ids) > 1 else 1
    train_segments = [segments[int(i)] for i in ids[:n_train]]
    eval_segments = [segments[int(i)] for i in ids[n_train:]] if len(ids) > 1 else [segments[0]]
    return train_segments, eval_segments


def get_u8(obs, key):
    if isinstance(obs, dict) and key in obs:
        x = np.asarray(obs[key], dtype=np.uint8)
    elif isinstance(obs, dict) and "rgb_obs" in obs and key in obs["rgb_obs"]:
        x = np.asarray(obs["rgb_obs"][key], dtype=np.uint8)
    else:
        raise KeyError("Could not find '{}' in observation".format(key))
    return x


def resize_if_needed(img, image_size=None):
    if image_size is None:
        return img
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


def sample_future_indices(cur_idx, goal_idx, num_future_frames):
    if goal_idx <= cur_idx:
        return np.full((num_future_frames,), cur_idx, dtype=np.int32)
    start = cur_idx + 1
    stop = goal_idx
    if start >= stop:
        return np.full((num_future_frames,), stop, dtype=np.int32)
    vals = np.linspace(start, stop, num=num_future_frames)
    vals = np.rint(vals).astype(np.int32)
    return np.clip(vals, start, stop)


def read_video_frames(path, count, image_size=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("future video not found: {}".format(path))
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if image_size is not None:
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.uint8))
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("no frames in future video: {}".format(path))
    ids = np.linspace(0, len(frames) - 1, num=count)
    ids = np.rint(ids).astype(np.int32)
    return np.stack([frames[int(i)] for i in ids], axis=0)


def make_null_future(future_static):
    future_static = np.asarray(future_static, dtype=np.uint8)
    return np.repeat(future_static[:1], repeats=future_static.shape[0], axis=0).astype(np.uint8)


def shift_future(future_static, shift):
    future_static = np.asarray(future_static, dtype=np.uint8)
    t = int(future_static.shape[0])
    idx = np.arange(t, dtype=np.int32) + int(shift)
    idx = np.clip(idx, 0, t - 1)
    return future_static[idx].astype(np.uint8)


def make_rafc_future_batch(future_static):
    futures = [make_null_future(future_static)]
    futures.extend(shift_future(future_static, shift) for shift in FUTURE_SHIFTS)
    return np.stack(futures, axis=0).astype(np.uint8)


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
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * ff_mult, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
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
        tok_static = self.encode_frames(obs_static)
        tok_gripper = self.encode_frames(obs_gripper)
        tok_future = self.encode_frames(future_static)
        x = torch.cat([tok_actions, tok_task, tok_goal, tok_static, tok_gripper, tok_future], dim=1)
        token_types = ([0] * self.chunk_horizon + [1] + [2] + [3] * self.obs_horizon + [4] * self.obs_horizon + [5] * self.future_horizon)
        token_types = torch.tensor(token_types, dtype=torch.long, device=x.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.modality_emb(token_types) + self.pos_emb[:, :x.shape[1], :]
        x = self.output_norm(self.transformer(self.input_norm(x)))
        future_start = self.chunk_horizon + 2 + self.obs_horizon + self.obs_horizon
        action_latent = x[:, :self.chunk_horizon]
        future_latent = x[:, future_start:future_start + self.future_horizon].mean(dim=1)
        return action_latent, future_latent

    def forward_with_latent(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        z, g_t = self.encode_tokens(obs_static, obs_gripper, future_static, task_id, goal_type)
        return z, g_t, self.arm_head(z), self.gripper_head(z).squeeze(-1)


class FrozenBCPolicy(object):
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.device = device
        self.tasks = list(cfg["tasks"])
        self.task_to_id = {t: i for i, t in enumerate(self.tasks)}
        self.arm_dim = int(cfg["arm_dim"])
        self.chunk_horizon = int(cfg["chunk_horizon"])
        self.obs_horizon = int(cfg["obs_horizon"])
        self.future_horizon = int(cfg["future_horizon"])
        self.arm_action_mean = np.asarray(ckpt["arm_action_mean"], dtype=np.float32)
        self.arm_action_std = np.asarray(ckpt["arm_action_std"], dtype=np.float32)
        self.model = FutureBC(
            arm_dim=self.arm_dim,
            chunk_horizon=self.chunk_horizon,
            obs_horizon=self.obs_horizon,
            future_horizon=self.future_horizon,
            num_tasks=int(cfg["num_tasks"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            num_heads=int(cfg["num_heads"]),
            dropout=float(cfg["dropout"]),
            ff_mult=int(cfg["ff_mult"]),
            pretrained_backbone=True,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()
        for pth in self.model.parameters():
            pth.requires_grad = False

    def task_id(self, task_name):
        return int(self.task_to_id[task_name])

    def extract_many(self, obs_static, obs_gripper, future_static_batch, task_name, goal_type=3):
        future_static_batch = np.asarray(future_static_batch, dtype=np.uint8)
        if future_static_batch.ndim == 4:
            future_static_batch = future_static_batch[None, ...]
        batch_size = int(future_static_batch.shape[0])
        obs_static_batch = np.repeat(np.asarray(obs_static, dtype=np.uint8)[None, ...], batch_size, axis=0)
        obs_gripper_batch = np.repeat(np.asarray(obs_gripper, dtype=np.uint8)[None, ...], batch_size, axis=0)

        obs_static_t = torch.from_numpy(obs_static_batch).to(self.device)
        obs_gripper_t = torch.from_numpy(obs_gripper_batch).to(self.device)
        future_static_t = torch.from_numpy(future_static_batch).to(self.device)
        task_t = torch.full((batch_size,), self.task_id(task_name), dtype=torch.long, device=self.device)
        goal_t = torch.full((batch_size,), int(goal_type), dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, g_t, arm_chunk_n, grip_logits = self.model.forward_with_latent(obs_static_t, obs_gripper_t, future_static_t, task_t, goal_t)

        feat = torch.cat([z[:, 0], z.mean(dim=1)], dim=-1).cpu().numpy().astype(np.float32)
        g_t = g_t.cpu().numpy().astype(np.float32)
        arm0_n = arm_chunk_n[:, 0].cpu().numpy().astype(np.float32)
        base_arm = arm0_n * self.arm_action_std[None, :] + self.arm_action_mean[None, :]
        grip = torch.sigmoid(grip_logits[:, 0]).cpu().numpy()
        base_grip = np.where(grip > 0.5, 1.0, -1.0).astype(np.float32).reshape(batch_size, 1)
        base_action = np.concatenate([base_arm[:, :self.arm_dim].astype(np.float32), base_grip], axis=1).astype(np.float32)
        return {"feature": feat, "g_t": g_t, "base_action": base_action}

    def extract(self, obs_static, obs_gripper, future_static, task_name, goal_type=3):
        batch = self.extract_many(obs_static, obs_gripper, np.asarray(future_static, dtype=np.uint8)[None, ...], task_name, goal_type)
        base_action = batch["base_action"][0]
        return {
            "feature": batch["feature"][0],
            "g_t": batch["g_t"][0],
            "base_arm": base_action[:self.arm_dim].astype(np.float32),
            "base_grip": float(base_action[self.arm_dim]),
        }

    def extract_with_future_latent(self, obs_static, obs_gripper, future_static, task_name, goal_type=3):
        base = self.extract(obs_static, obs_gripper, future_static, task_name, goal_type)
        base_action = np.concatenate([base["base_arm"][:self.arm_dim], np.asarray([base["base_grip"]], dtype=np.float32)], axis=0).astype(np.float32)
        return base["feature"], base["g_t"], base_action


class FineTuningActor(nn.Module):
    def __init__(self, feature_dim, g_dim, action_dim, arm_dim, arm_limit, grip_limit):
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.action_dim = int(action_dim)
        self.g_dim = int(g_dim)
        self.register_buffer("limit_vec", torch.tensor([arm_limit] * arm_dim + [grip_limit], dtype=torch.float32).view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + self.g_dim + int(action_dim), 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, g_t, base_action):
        x = torch.cat([feature, g_t, base_action], dim=-1)
        return torch.tanh(self.net(x)) * self.limit_vec.to(x.device)


class FutureGate(nn.Module):
    def __init__(self, feature_dim, num_shifts):
        super().__init__()
        self.num_shifts = int(num_shifts)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) * (self.num_shifts + 1), 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.alpha_head = nn.Linear(128, 1)
        self.shift_head = nn.Linear(128, self.num_shifts)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, float(np.log(RAFC_INIT_ALPHA / (1.0 - RAFC_INIT_ALPHA))))
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        self.shift_head.bias.data[self.num_shifts // 2] = RAFC_CENTER_LOGIT_BIAS

    def forward(self, feature_null, feature_candidates):
        batch_size, num_shifts, feature_dim = feature_candidates.shape
        x = torch.cat([feature_null, feature_candidates.reshape(batch_size, num_shifts * feature_dim)], dim=-1)
        h = self.net(x)
        alpha = torch.sigmoid(self.alpha_head(h))
        weights = torch.softmax(self.shift_head(h), dim=-1)
        feature_shift = torch.sum(weights.unsqueeze(-1) * feature_candidates, dim=1)
        feature_gate = (1.0 - alpha) * feature_null + alpha * feature_shift
        return feature_gate, alpha, weights


def mix_with_gate(alpha, weights, value_null, value_candidates):
    value_shift = torch.sum(weights.unsqueeze(-1) * value_candidates, dim=1)
    return (1.0 - alpha) * value_null + alpha * value_shift


class CriticQ(nn.Module):
    def __init__(self, feature_dim, g_dim, priv_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + int(g_dim) + int(priv_dim) + int(action_dim) * 2, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, feature, g_t, priv_state, base_action, delta_action):
        return self.net(torch.cat([feature, g_t, priv_state, base_action, delta_action], dim=-1))


class TD3FineTuningAgent(object):
    def __init__(self, feature_dim, g_dim, priv_dim, action_dim, arm_dim, device, use_rafc=True):
        self.device = device
        self.feature_dim = int(feature_dim)
        self.g_dim = int(g_dim)
        self.priv_dim = int(priv_dim)
        self.action_dim = int(action_dim)
        self.arm_dim = int(arm_dim)
        self.use_rafc = bool(use_rafc)
        self.center_shift_idx = FUTURE_SHIFTS.index(0) if 0 in FUTURE_SHIFTS else len(FUTURE_SHIFTS) // 2
        self.actor = FineTuningActor(feature_dim, g_dim, action_dim, arm_dim, DELTA_ARM_LIMIT, GRIP_DELTA_LIMIT).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.gate = FutureGate(feature_dim, len(FUTURE_SHIFTS)).to(device) if self.use_rafc else None
        self.gate_target = deepcopy(self.gate).to(device) if self.use_rafc else None
        self.critic1 = CriticQ(feature_dim, g_dim, priv_dim, action_dim).to(device)
        self.critic2 = CriticQ(feature_dim, g_dim, priv_dim, action_dim).to(device)
        self.critic1_target = deepcopy(self.critic1).to(device)
        self.critic2_target = deepcopy(self.critic2).to(device)
        self.actor_params = list(self.actor.parameters()) + (list(self.gate.parameters()) if self.use_rafc else [])
        self.actor_opt = torch.optim.AdamW(self.actor_params, lr=ACTOR_LR, weight_decay=WEIGHT_DECAY)
        self.critic_opt = torch.optim.AdamW(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        self.total_updates = 0
        self.limit_vec_np = np.asarray([DELTA_ARM_LIMIT] * arm_dim + [GRIP_DELTA_LIMIT], dtype=np.float32)

    def _apply_gate(self, feature_null, feature_candidates, g_null, g_candidates, base_action_null, base_action_candidates, target=False):
        if not self.use_rafc:
            batch_size = int(feature_candidates.shape[0])
            weights = torch.zeros((batch_size, len(FUTURE_SHIFTS)), dtype=feature_candidates.dtype, device=feature_candidates.device)
            weights[:, self.center_shift_idx] = 1.0
            alpha = torch.ones((batch_size, 1), dtype=feature_candidates.dtype, device=feature_candidates.device)
            return (
                feature_candidates[:, self.center_shift_idx],
                g_candidates[:, self.center_shift_idx],
                base_action_candidates[:, self.center_shift_idx],
                alpha,
                weights,
            )
        gate = self.gate_target if target else self.gate
        feature_gate, alpha, weights = gate(feature_null, feature_candidates)
        g_gate = mix_with_gate(alpha, weights, g_null, g_candidates)
        base_action_gate = mix_with_gate(alpha, weights, base_action_null, base_action_candidates)
        return feature_gate, g_gate, base_action_gate, alpha, weights

    def act(self, feature_null_np, g_null_np, base_action_null_np, feature_candidates_np, g_candidates_np, base_action_candidates_np, add_noise=True, return_gate=False):
        feature_null = torch.from_numpy(feature_null_np).unsqueeze(0).to(self.device)
        g_null = torch.from_numpy(g_null_np).unsqueeze(0).to(self.device)
        base_action_null = torch.from_numpy(base_action_null_np).unsqueeze(0).to(self.device)
        feature_candidates = torch.from_numpy(feature_candidates_np).unsqueeze(0).to(self.device)
        g_candidates = torch.from_numpy(g_candidates_np).unsqueeze(0).to(self.device)
        base_action_candidates = torch.from_numpy(base_action_candidates_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature, g_t, base_action, alpha, weights = self._apply_gate(
                feature_null,
                feature_candidates,
                g_null,
                g_candidates,
                base_action_null,
                base_action_candidates,
            )
            delta = self.actor(feature, g_t, base_action)[0].cpu().numpy()
            base_action_np = base_action[0].cpu().numpy().astype(np.float32)
            alpha_np = alpha[0].cpu().numpy().astype(np.float32)
            weights_np = weights[0].cpu().numpy().astype(np.float32)
        if add_noise:
            noise = np.random.normal(0.0, EXPL_NOISE_STD, size=delta.shape).astype(np.float32)
            delta = delta + np.clip(noise, -EXPL_NOISE_CLIP, EXPL_NOISE_CLIP)
        delta = np.clip(delta, -self.limit_vec_np, self.limit_vec_np).astype(np.float32)
        if return_gate:
            return delta, {"base_action": base_action_np, "alpha": alpha_np, "weights": weights_np}
        return delta

    def gated_base_action(self, feature_null_np, g_null_np, base_action_null_np, feature_candidates_np, g_candidates_np, base_action_candidates_np):
        feature_null = torch.from_numpy(feature_null_np).unsqueeze(0).to(self.device)
        g_null = torch.from_numpy(g_null_np).unsqueeze(0).to(self.device)
        base_action_null = torch.from_numpy(base_action_null_np).unsqueeze(0).to(self.device)
        feature_candidates = torch.from_numpy(feature_candidates_np).unsqueeze(0).to(self.device)
        g_candidates = torch.from_numpy(g_candidates_np).unsqueeze(0).to(self.device)
        base_action_candidates = torch.from_numpy(base_action_candidates_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, base_action, alpha, weights = self._apply_gate(
                feature_null,
                feature_candidates,
                g_null,
                g_candidates,
                base_action_null,
                base_action_candidates,
            )
        return base_action[0].cpu().numpy().astype(np.float32), {
            "alpha": alpha[0].cpu().numpy().astype(np.float32),
            "weights": weights[0].cpu().numpy().astype(np.float32),
        }

    def train_step(self, replay, batch_size):
        batch = replay.sample(batch_size)
        feature_null = torch.from_numpy(batch["feature_null"]).to(self.device)
        feature_candidates = torch.from_numpy(batch["feature_candidates"]).to(self.device)
        g_null = torch.from_numpy(batch["g_null"]).to(self.device)
        g_candidates = torch.from_numpy(batch["g_candidates"]).to(self.device)
        priv_state = torch.from_numpy(batch["priv_state"]).to(self.device)
        base_action_null = torch.from_numpy(batch["base_action_null"]).to(self.device)
        base_action_candidates = torch.from_numpy(batch["base_action_candidates"]).to(self.device)
        delta_action = torch.from_numpy(batch["delta_action"]).to(self.device)
        reward = torch.from_numpy(batch["reward"]).to(self.device)
        next_feature_null = torch.from_numpy(batch["next_feature_null"]).to(self.device)
        next_feature_candidates = torch.from_numpy(batch["next_feature_candidates"]).to(self.device)
        next_g_null = torch.from_numpy(batch["next_g_null"]).to(self.device)
        next_g_candidates = torch.from_numpy(batch["next_g_candidates"]).to(self.device)
        next_priv_state = torch.from_numpy(batch["next_priv_state"]).to(self.device)
        next_base_action_null = torch.from_numpy(batch["next_base_action_null"]).to(self.device)
        next_base_action_candidates = torch.from_numpy(batch["next_base_action_candidates"]).to(self.device)
        done = torch.from_numpy(batch["done"]).to(self.device)
        with torch.no_grad():
            feature, g_t, base_action, _, _ = self._apply_gate(
                feature_null,
                feature_candidates,
                g_null,
                g_candidates,
                base_action_null,
                base_action_candidates,
            )
            next_feature, next_g, next_base_action, _, _ = self._apply_gate(
                next_feature_null,
                next_feature_candidates,
                next_g_null,
                next_g_candidates,
                next_base_action_null,
                next_base_action_candidates,
                target=True,
            )
            noise = torch.randn_like(delta_action) * POLICY_NOISE
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            limit_vec = self.actor.limit_vec.to(self.device)
            next_delta = torch.clamp(self.actor_target(next_feature, next_g, next_base_action) + noise, -limit_vec, limit_vec)
            q1_t = self.critic1_target(next_feature, next_g, next_priv_state, next_base_action, next_delta)
            q2_t = self.critic2_target(next_feature, next_g, next_priv_state, next_base_action, next_delta)
            target = reward + (1.0 - done) * DISCOUNT * torch.min(q1_t, q2_t)
        q1 = self.critic1(feature, g_t, priv_state, base_action, delta_action)
        q2 = self.critic2(feature, g_t, priv_state, base_action, delta_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=5.0)
        self.critic_opt.step()
        logs = {"critic_loss": float(critic_loss.item()), "q1": float(q1.mean().item())}
        self.total_updates += 1
        if self.total_updates % POLICY_DELAY == 0:
            feature_pi, g_pi, base_action_pi, alpha, weights = self._apply_gate(
                feature_null,
                feature_candidates,
                g_null,
                g_candidates,
                base_action_null,
                base_action_candidates,
            )
            delta = self.actor(feature_pi, g_pi, base_action_pi)
            actor_loss = -self.critic1(feature_pi, g_pi, priv_state, base_action_pi, delta).mean()
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            gate_reg = RAFC_ALPHA_REG * (alpha.mean() - RAFC_ALPHA_TARGET).pow(2) - RAFC_ENTROPY_REG * entropy
            actor_loss = actor_loss + gate_reg
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, max_norm=5.0)
            self.actor_opt.step()
            self.soft_update(self.actor, self.actor_target)
            if self.use_rafc:
                self.soft_update(self.gate, self.gate_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            logs["actor_loss"] = float(actor_loss.item())
            logs["gate_alpha"] = float(alpha.mean().item())
            logs["gate_entropy"] = float(entropy.item())
            logs["gate_weights"] = [float(x) for x in weights.mean(dim=0).detach().cpu().numpy()]
        return logs

    def soft_update(self, src, tgt):
        with torch.no_grad():
            for a, b in zip(src.parameters(), tgt.parameters()):
                b.data.mul_(1.0 - TAU).add_(TAU * a.data)

    def save(self, path):
        ckpt = {
            "actor_state_dict": self.actor.state_dict(),
            "uses_rafc": self.use_rafc,
            "future_shifts": list(FUTURE_SHIFTS),
            "num_future_shifts": len(FUTURE_SHIFTS),
            "feature_dim": self.feature_dim,
            "g_dim": self.g_dim,
            "priv_dim": self.priv_dim,
            "action_dim": self.action_dim,
            "arm_dim": self.arm_dim,
            "delta_arm_limit": DELTA_ARM_LIMIT,
            "grip_delta_limit": GRIP_DELTA_LIMIT,
        }
        if self.use_rafc:
            ckpt["gate_state_dict"] = self.gate.state_dict()
        torch.save(ckpt, path)


class ReplayBuffer(object):
    def __init__(self, capacity, feature_dim, g_dim, priv_dim, action_dim, num_shifts):
        self.capacity = int(capacity)
        self.num_shifts = int(num_shifts)
        self.feature_null = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.feature_candidates = np.zeros((capacity, self.num_shifts, feature_dim), dtype=np.float32)
        self.g_null = np.zeros((capacity, g_dim), dtype=np.float32)
        self.g_candidates = np.zeros((capacity, self.num_shifts, g_dim), dtype=np.float32)
        self.priv_state = np.zeros((capacity, priv_dim), dtype=np.float32)
        self.base_action_null = np.zeros((capacity, action_dim), dtype=np.float32)
        self.base_action_candidates = np.zeros((capacity, self.num_shifts, action_dim), dtype=np.float32)
        self.delta_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_feature_null = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.next_feature_candidates = np.zeros((capacity, self.num_shifts, feature_dim), dtype=np.float32)
        self.next_g_null = np.zeros((capacity, g_dim), dtype=np.float32)
        self.next_g_candidates = np.zeros((capacity, self.num_shifts, g_dim), dtype=np.float32)
        self.next_priv_state = np.zeros((capacity, priv_dim), dtype=np.float32)
        self.next_base_action_null = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_base_action_candidates = np.zeros((capacity, self.num_shifts, action_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, rafc, priv_state, delta_action, reward, next_rafc, next_priv_state, done):
        i = self.ptr
        self.feature_null[i] = rafc["feature_null"]
        self.feature_candidates[i] = rafc["feature_candidates"]
        self.g_null[i] = rafc["g_null"]
        self.g_candidates[i] = rafc["g_candidates"]
        self.priv_state[i] = priv_state
        self.base_action_null[i] = rafc["base_action_null"]
        self.base_action_candidates[i] = rafc["base_action_candidates"]
        self.delta_action[i] = delta_action
        self.reward[i, 0] = float(reward)
        self.next_feature_null[i] = next_rafc["feature_null"]
        self.next_feature_candidates[i] = next_rafc["feature_candidates"]
        self.next_g_null[i] = next_rafc["g_null"]
        self.next_g_candidates[i] = next_rafc["g_candidates"]
        self.next_priv_state[i] = next_priv_state
        self.next_base_action_null[i] = next_rafc["base_action_null"]
        self.next_base_action_candidates[i] = next_rafc["base_action_candidates"]
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "feature_null": self.feature_null[idx],
            "feature_candidates": self.feature_candidates[idx],
            "g_null": self.g_null[idx],
            "g_candidates": self.g_candidates[idx],
            "priv_state": self.priv_state[idx],
            "base_action_null": self.base_action_null[idx],
            "base_action_candidates": self.base_action_candidates[idx],
            "delta_action": self.delta_action[idx],
            "reward": self.reward[idx],
            "next_feature_null": self.next_feature_null[idx],
            "next_feature_candidates": self.next_feature_candidates[idx],
            "next_g_null": self.next_g_null[idx],
            "next_g_candidates": self.next_g_candidates[idx],
            "next_priv_state": self.next_priv_state[idx],
            "next_base_action_null": self.next_base_action_null[idx],
            "next_base_action_candidates": self.next_base_action_candidates[idx],
            "done": self.done[idx],
        }


class CalvinFineTuningEnv(object):
    def __init__(self, segments, show_gui, seed, base_policy, cache, fixed_order=False):
        self.segments = list(segments)
        self.show_gui = bool(show_gui)
        self.rng = np.random.default_rng(seed)
        self.base_policy = base_policy
        self.cache = cache
        self.fixed_order = bool(fixed_order)
        self.ptr = 0
        self.env = None
        self.tasks_oracle = None
        self.current_segment = None
        self.start_info = None
        self.last_obs = None
        self.episode_step = 0
        self.obs_static_hist = deque(maxlen=self.base_policy.obs_horizon)
        self.obs_gripper_hist = deque(maxlen=self.base_policy.obs_horizon)
        self.prev_aligned_idx = None
        self.prev_raw_idx = None
        self.prev_priv_state = None

    def _ensure_env(self):
        if self.env is None or self.tasks_oracle is None:
            self.env, self.tasks_oracle = make_env(self.show_gui)

    def _pick_segment(self):
        if self.fixed_order:
            seg = self.segments[self.ptr % len(self.segments)]
            self.ptr += 1
            return seg
        return self.segments[int(self.rng.integers(0, len(self.segments)))]

    def _phase_ratio(self, idx):
        s = int(self.current_segment["global_start_idx"])
        e = int(self.current_segment["global_end_idx"])
        return float(np.clip((int(idx) - s) / float(max(e - s, 1)), 0.0, 1.0))

    def _nearest_progress_index(self, robot_obs, scene_obs):
        seg_start = int(self.current_segment["global_start_idx"])
        seg_end = int(self.current_segment["global_end_idx"])
        anchor = seg_start if self.episode_step < PHASE_INIT_STEPS else int(self.prev_raw_idx or seg_start)
        lo = max(seg_start, anchor - ALIGN_BACKWARD_SEARCH)
        hi = min(seg_end, anchor + ALIGN_FORWARD_SEARCH)
        cur = np.concatenate([np.asarray(robot_obs, dtype=np.float32), np.asarray(scene_obs, dtype=np.float32)], axis=0)
        best_idx = lo
        best_dist = None
        for j in range(lo, hi + 1):
            item = self.cache.get(j)
            ref = np.concatenate([np.asarray(item["robot_obs"], dtype=np.float32), np.asarray(item["scene_obs"], dtype=np.float32)], axis=0)
            d = float(np.mean(np.square(cur - ref)))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = j
        prev = int(self.prev_aligned_idx if self.prev_aligned_idx is not None else seg_start)
        best_idx = int(np.clip(best_idx, prev - ALIGN_MAX_BACKWARD_STEP, prev + ALIGN_MAX_FORWARD_STEP))
        best_idx = int(np.clip(best_idx, seg_start, seg_end))
        return best_idx, float(best_dist if best_dist is not None else 0.0)

    def _future_from_demo_or_video(self, aligned_idx):
        task = self.current_segment["task"]
        if USE_GENERATED_FUTURES_DURING_RL:
            pth = GENERATED_FUTURE_ROOT / task / "inpainted_robot_future.mp4"
            if pth.exists():
                return read_video_frames(pth, self.base_policy.future_horizon, IMAGE_SIZE)
        seg_end = int(self.current_segment["global_end_idx"])
        fut_idx = sample_future_indices(aligned_idx, seg_end, self.base_policy.future_horizon)
        return np.stack([resize_if_needed(np.asarray(self.cache.get(int(j))["rgb_static"], dtype=np.uint8), IMAGE_SIZE) for j in fut_idx], axis=0)

    def _policy_input(self, obs):
        robot = np.asarray(obs["robot_obs"], dtype=np.float32)
        scene = np.asarray(obs["scene_obs"], dtype=np.float32)
        aligned_idx, dist = self._nearest_progress_index(robot, scene)
        self.prev_raw_idx = aligned_idx
        future_static = self._future_from_demo_or_video(aligned_idx)
        return {
            "obs_static": np.stack(list(self.obs_static_hist), axis=0).astype(np.uint8),
            "obs_gripper": np.stack(list(self.obs_gripper_hist), axis=0).astype(np.uint8),
            "future_static": future_static.astype(np.uint8),
            "task": self.current_segment["task"],
            "goal_type": 3,
            "aligned_idx": int(aligned_idx),
            "aligned_dist": float(dist),
            "priv_state": np.concatenate([robot, scene], axis=0).astype(np.float32),
        }

    def reset(self, seed=None):
        self._ensure_env()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_segment = self._pick_segment()
        self.episode_step = 0
        start_idx = int(self.current_segment["global_start_idx"])
        start_item = self.cache.get(start_idx)
        obs = self.env.reset(robot_obs=np.asarray(start_item["robot_obs"], dtype=np.float32), scene_obs=np.asarray(start_item["scene_obs"], dtype=np.float32))
        self.start_info = self.env.get_info()
        self.last_obs = obs
        self.obs_static_hist.clear()
        self.obs_gripper_hist.clear()
        init_static = resize_if_needed(get_u8(obs, "rgb_static"), IMAGE_SIZE)
        init_gripper = resize_if_needed(get_u8(obs, "rgb_gripper"), IMAGE_SIZE)
        for _ in range(self.base_policy.obs_horizon):
            self.obs_static_hist.append(init_static.copy())
            self.obs_gripper_hist.append(init_gripper.copy())
        self.prev_aligned_idx = start_idx
        self.prev_raw_idx = start_idx
        pi = self._policy_input(obs)
        self.prev_aligned_idx = int(pi["aligned_idx"])
        self.prev_priv_state = pi["priv_state"].copy()
        return pi, {"task": self.current_segment["task"], "segment_id": int(self.current_segment["segment_id"])}

    def _compose_full_action(self, base_action, delta_action):
        arm = np.clip(base_action[:ARM_ACTION_DIM] + delta_action[:ARM_ACTION_DIM], -1.0, 1.0).astype(np.float32)
        grip_score = float(base_action[ARM_ACTION_DIM] + delta_action[ARM_ACTION_DIM])
        grip = 1.0 if grip_score >= 0.0 else -1.0
        return np.concatenate([arm, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)

    def _reward(self, prev_idx, new_idx, prev_priv, new_priv, delta, success, truncated):
        seg_start = int(self.current_segment["global_start_idx"])
        seg_end = int(self.current_segment["global_end_idx"])
        seg_len = max(seg_end - seg_start, 1)
        target_idx = min(int(prev_idx) + 5, seg_end)
        target_item = self.cache.get(target_idx)
        robot_dim = len(target_item["robot_obs"])
        target_priv = np.concatenate([np.asarray(target_item["robot_obs"], dtype=np.float32), np.asarray(target_item["scene_obs"], dtype=np.float32)], axis=0)
        prev_robot, prev_scene = prev_priv[:robot_dim], prev_priv[robot_dim:]
        new_robot, new_scene = new_priv[:robot_dim], new_priv[robot_dim:]
        tgt_robot, tgt_scene = target_priv[:robot_dim], target_priv[robot_dim:]
        prev_rd = float(np.mean(np.square(prev_robot - tgt_robot)))
        new_rd = float(np.mean(np.square(new_robot - tgt_robot)))
        prev_sd = float(np.mean(np.square(prev_scene - tgt_scene)))
        new_sd = float(np.mean(np.square(new_scene - tgt_scene)))
        index_delta = float(np.clip((int(new_idx) - int(prev_idx)) / float(seg_len), -0.25, 0.25))
        reward = STEP_PENALTY
        reward += INDEX_PROGRESS_SCALE * max(0.0, index_delta)
        reward += ROBOT_PROGRESS_SCALE * max(0.0, prev_rd - new_rd)
        reward += SCENE_PROGRESS_SCALE * max(0.0, prev_sd - new_sd)
        reward -= DELTA_L2 * float(np.mean(np.square(delta[:ARM_ACTION_DIM])))
        if success:
            reward += SUCCESS_REWARD
        elif truncated:
            reward += TIMEOUT_PENALTY
        return float(np.clip(reward, -50.0, 150.0))

    def step(self, full_action, delta_action):
        broken = False
        try:
            step_res = self.env.step(np.asarray(full_action, dtype=np.float32))
        except AssertionError:
            broken = True
            step_res = None
        self.episode_step += 1
        if broken or step_res is None:
            return None, BROKEN_PENALTY, False, True, {"success": False, "broken": True, "task": self.current_segment["task"]}
        if len(step_res) == 5:
            obs, _, terminated_raw, truncated_raw, _ = step_res
            env_done = bool(terminated_raw) or bool(truncated_raw)
        else:
            obs, _, env_done, _ = step_res
        self.last_obs = obs
        self.obs_static_hist.append(resize_if_needed(get_u8(obs, "rgb_static"), IMAGE_SIZE))
        self.obs_gripper_hist.append(resize_if_needed(get_u8(obs, "rgb_gripper"), IMAGE_SIZE))
        curr_info = self.env.get_info()
        task = self.current_segment["task"]
        success = oracle_success(self.tasks_oracle, self.start_info, curr_info, task)
        next_pi = self._policy_input(obs)
        new_idx = int(next_pi["aligned_idx"])
        new_priv = next_pi["priv_state"].copy()
        truncated = bool(env_done) or self.episode_step >= MAX_EPISODE_STEPS
        reward = self._reward(self.prev_aligned_idx, new_idx, self.prev_priv_state, new_priv, delta_action, success, truncated)
        self.prev_aligned_idx = new_idx
        self.prev_priv_state = new_priv.copy()
        info = {"success": bool(success), "broken": False, "task": task, "segment_id": int(self.current_segment["segment_id"]), "phase": self._phase_ratio(new_idx), "episode_step": int(self.episode_step)}
        return next_pi, reward, bool(success), bool(truncated), info

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass
        self.env = None
        self.tasks_oracle = None


def to_base_action(base):
    return np.concatenate([base["base_arm"][:ARM_ACTION_DIM], np.asarray([base["base_grip"]], dtype=np.float32)], axis=0).astype(np.float32)


def extract_rafc_inputs(base_policy, pi):
    future_batch = make_rafc_future_batch(pi["future_static"])
    batch = base_policy.extract_many(
        pi["obs_static"],
        pi["obs_gripper"],
        future_batch,
        pi["task"],
        pi["goal_type"],
    )
    return {
        "feature_null": batch["feature"][0].astype(np.float32),
        "feature_candidates": batch["feature"][1:].astype(np.float32),
        "g_null": batch["g_t"][0].astype(np.float32),
        "g_candidates": batch["g_t"][1:].astype(np.float32),
        "base_action_null": batch["base_action"][0].astype(np.float32),
        "base_action_candidates": batch["base_action"][1:].astype(np.float32),
    }


def eval_agent(env, base_policy, agent, num_eps):
    successes = 0
    rows = []
    for ep in range(num_eps):
        pi, info = env.reset(seed=SEED + 30_000 + ep)
        done = False
        trunc = False
        step = 0
        last = info
        while not done and not trunc:
            rafc = extract_rafc_inputs(base_policy, pi)
            delta, gate_info = agent.act(
                rafc["feature_null"],
                rafc["g_null"],
                rafc["base_action_null"],
                rafc["feature_candidates"],
                rafc["g_candidates"],
                rafc["base_action_candidates"],
                add_noise=False,
                return_gate=True,
            )
            base_action = gate_info["base_action"]
            full_action = env._compose_full_action(base_action, delta)
            pi, _, done, trunc, last = env.step(full_action, delta)
            step += 1
            if pi is None:
                break
        successes += int(bool(last.get("success", False)))
        rows.append({"episode": ep + 1, "task": last.get("task", ""), "success": bool(last.get("success", False)), "steps": step})
    return float(successes / max(num_eps, 1)), rows


def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)
    if DATA_ROOT is None:
        raise RuntimeError("Set CALVIN_DATA_ROOT or CALVIN_ROOT before RL fine-tuning")
    if not DATA_ROOT.exists():
        raise FileNotFoundError("Missing DATA_ROOT: {}".format(DATA_ROOT))
    if not BC_CKPT_PATH.exists():
        raise FileNotFoundError("Missing BC checkpoint: {}".format(BC_CKPT_PATH))

    segments, tasks, task_to_id = load_segments(SEGMENTS_JSON)
    train_segments, eval_segments = split_segments(segments, TRAIN_SPLIT, SEED)
    cache = EpisodeCache(DATA_ROOT)
    base_policy = FrozenBCPolicy(BC_CKPT_PATH, DEVICE)
    train_env = CalvinFineTuningEnv(train_segments, SHOW_GUI_TRAIN, SEED, base_policy, cache, fixed_order=False)
    eval_env = CalvinFineTuningEnv(eval_segments, SHOW_GUI_EVAL, SEED + 999, base_policy, cache, fixed_order=True)

    pi, info = train_env.reset(seed=SEED)
    rafc = extract_rafc_inputs(base_policy, pi)
    feature_dim = int(rafc["feature_null"].shape[0])
    g_dim = int(rafc["g_null"].shape[0])
    priv_dim = int(pi["priv_state"].shape[0])
    replay = ReplayBuffer(BUFFER_SIZE, feature_dim, g_dim, priv_dim, ACTION_DIM, len(FUTURE_SHIFTS))
    agent = TD3FineTuningAgent(feature_dim, g_dim, priv_dim, ACTION_DIM, ARM_ACTION_DIM, DEVICE, use_rafc=USE_RAFC)
    print("future_source={}".format("generated" if USE_GENERATED_FUTURES_DURING_RL else "demo"))
    print("use_rafc={} future_shifts={}".format(bool(USE_RAFC), list(FUTURE_SHIFTS)))

    history = []
    best_success = -1.0
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    recent_success = deque(maxlen=20)

    for env_step in range(1, TOTAL_ENV_STEPS + 1):
        if env_step <= RANDOM_WARMUP_STEPS:
            base_action, _ = agent.gated_base_action(
                rafc["feature_null"],
                rafc["g_null"],
                rafc["base_action_null"],
                rafc["feature_candidates"],
                rafc["g_candidates"],
                rafc["base_action_candidates"],
            )
            limit_vec = np.asarray([DELTA_ARM_LIMIT] * ARM_ACTION_DIM + [GRIP_DELTA_LIMIT], dtype=np.float32)
            delta = np.random.uniform(low=-limit_vec, high=limit_vec).astype(np.float32)
        else:
            delta, gate_info = agent.act(
                rafc["feature_null"],
                rafc["g_null"],
                rafc["base_action_null"],
                rafc["feature_candidates"],
                rafc["g_candidates"],
                rafc["base_action_candidates"],
                add_noise=True,
                return_gate=True,
            )
            base_action = gate_info["base_action"]
        full_action = train_env._compose_full_action(base_action, delta)
        next_pi, reward, done, truncated, step_info = train_env.step(full_action, delta)
        terminal = bool(done or truncated)

        if next_pi is None:
            next_rafc = rafc
            next_priv = pi["priv_state"].copy()
        else:
            next_rafc = extract_rafc_inputs(base_policy, next_pi)
            next_priv = next_pi["priv_state"].copy()

        replay.add(rafc, pi["priv_state"], delta, reward, next_rafc, next_priv, float(terminal))
        episode_reward += float(reward)
        episode_steps += 1

        if replay.size >= START_TRAIN_AFTER:
            logs = agent.train_step(replay, BATCH_SIZE)
        else:
            logs = {}

        if terminal:
            episode_count += 1
            recent_success.append(float(step_info.get("success", False)))
            print("ep {:04d} step {} task={} return={:.2f} success={} steps={}".format(episode_count, env_step, step_info.get("task", ""), episode_reward, bool(step_info.get("success", False)), episode_steps))
            pi, info = train_env.reset(seed=SEED + episode_count)
            rafc = extract_rafc_inputs(base_policy, pi)
            episode_reward = 0.0
            episode_steps = 0
        else:
            pi = next_pi
            rafc = next_rafc

        if env_step % EVAL_EVERY_STEPS == 0:
            rate, rows = eval_agent(eval_env, base_policy, agent, NUM_EVAL_EPISODES)
            row = {"env_step": env_step, "success_rate": rate, "recent_success": float(np.mean(recent_success)) if len(recent_success) else 0.0, "eval": rows, "logs": logs}
            history.append(row)
            with open(HISTORY_JSON, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            print("eval step {} success {:.3f}".format(env_step, rate))
            if rate > best_success:
                best_success = rate
                agent.save(BEST_ACTOR_PATH)

    agent.save(FINAL_ACTOR_PATH)
    train_env.close()
    eval_env.close()
    try:
        p.disconnect()
    except Exception:
        pass
    print("done", FINAL_ACTOR_PATH)


if __name__ == "__main__":
    main()
