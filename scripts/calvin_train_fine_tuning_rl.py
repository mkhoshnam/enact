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


CALVIN_ROOT = Path("/path/to/calvin")
DATA_ROOT = CALVIN_ROOT / "dataset/task_D_D/training"
OUT_BASE = Path("/path/to/enact_calvin_outputs")
SEGMENTS_JSON = OUT_BASE / "calvin" / "segments_future_bc.json"
BC_CKPT_PATH = OUT_BASE / "calvin_bc" / "bc_actor_best.pt"
RESULTS_DIR = OUT_BASE / "calvin_fine_tuning_rl"
FINAL_ACTOR_PATH = RESULTS_DIR / "fine_tuning_actor_final.pt"
BEST_ACTOR_PATH = RESULTS_DIR / "fine_tuning_actor_best.pt"
HISTORY_JSON = RESULTS_DIR / "history.json"

GENERATED_FUTURE_ROOT = Path("/path/to/generated_inpainted_calvin_futures")
USE_GENERATED_FUTURES_DURING_RL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
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


def load_segments(path):
    if not path.exists():
        raise FileNotFoundError("Missing segments json: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["segments"], obj["tasks"], obj.get("task_to_id", {t: i for i, t in enumerate(obj["tasks"])})


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
        x = self.transformer(self.input_norm(x))
        return self.output_norm(x[:, :self.chunk_horizon])

    def forward_with_latent(self, obs_static, obs_gripper, future_static, task_id, goal_type):
        z = self.encode_tokens(obs_static, obs_gripper, future_static, task_id, goal_type)
        return z, self.arm_head(z), self.gripper_head(z).squeeze(-1)


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

    def extract(self, obs_static, obs_gripper, future_static, task_name, goal_type=3):
        obs_static_t = torch.from_numpy(obs_static).unsqueeze(0).to(self.device)
        obs_gripper_t = torch.from_numpy(obs_gripper).unsqueeze(0).to(self.device)
        future_static_t = torch.from_numpy(future_static).unsqueeze(0).to(self.device)
        task_t = torch.tensor([self.task_id(task_name)], dtype=torch.long, device=self.device)
        goal_t = torch.tensor([int(goal_type)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, arm_chunk_n, grip_logits = self.model.forward_with_latent(obs_static_t, obs_gripper_t, future_static_t, task_t, goal_t)
        z = z[0]
        feat = torch.cat([z[0], z.mean(dim=0)], dim=-1).cpu().numpy().astype(np.float32)
        arm_chunk_n = arm_chunk_n[0].cpu().numpy().astype(np.float32)
        arm_chunk = arm_chunk_n * self.arm_action_std[None, :] + self.arm_action_mean[None, :]
        base_grip = 1.0 if float(torch.sigmoid(grip_logits[0, 0]).item()) > 0.5 else -1.0
        return {"feature": feat, "base_arm": arm_chunk[0].astype(np.float32), "base_grip": float(base_grip)}


class FineTuningActor(nn.Module):
    def __init__(self, feature_dim, action_dim, arm_dim, arm_limit, grip_limit):
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.action_dim = int(action_dim)
        self.register_buffer("limit_vec", torch.tensor([arm_limit] * arm_dim + [grip_limit], dtype=torch.float32).view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + int(action_dim), 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature, base_action):
        x = torch.cat([feature, base_action], dim=-1)
        return torch.tanh(self.net(x)) * self.limit_vec.to(x.device)


class CriticQ(nn.Module):
    def __init__(self, feature_dim, priv_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + int(priv_dim) + int(action_dim) * 2, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, feature, priv_state, base_action, delta_action):
        return self.net(torch.cat([feature, priv_state, base_action, delta_action], dim=-1))


class TD3FineTuningAgent(object):
    def __init__(self, feature_dim, priv_dim, action_dim, arm_dim, device):
        self.device = device
        self.feature_dim = int(feature_dim)
        self.priv_dim = int(priv_dim)
        self.action_dim = int(action_dim)
        self.arm_dim = int(arm_dim)
        self.actor = FineTuningActor(feature_dim, action_dim, arm_dim, DELTA_ARM_LIMIT, GRIP_DELTA_LIMIT).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.critic1 = CriticQ(feature_dim, priv_dim, action_dim).to(device)
        self.critic2 = CriticQ(feature_dim, priv_dim, action_dim).to(device)
        self.critic1_target = deepcopy(self.critic1).to(device)
        self.critic2_target = deepcopy(self.critic2).to(device)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=ACTOR_LR, weight_decay=WEIGHT_DECAY)
        self.critic_opt = torch.optim.AdamW(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY)
        self.total_updates = 0
        self.limit_vec_np = np.asarray([DELTA_ARM_LIMIT] * arm_dim + [GRIP_DELTA_LIMIT], dtype=np.float32)

    def act(self, feature_np, base_action_np, add_noise=True):
        feature = torch.from_numpy(feature_np).unsqueeze(0).to(self.device)
        base_action = torch.from_numpy(base_action_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.actor(feature, base_action)[0].cpu().numpy()
        if add_noise:
            noise = np.random.normal(0.0, EXPL_NOISE_STD, size=delta.shape).astype(np.float32)
            delta = delta + np.clip(noise, -EXPL_NOISE_CLIP, EXPL_NOISE_CLIP)
        return np.clip(delta, -self.limit_vec_np, self.limit_vec_np).astype(np.float32)

    def train_step(self, replay, batch_size):
        batch = replay.sample(batch_size)
        feature = torch.from_numpy(batch["feature"]).to(self.device)
        priv_state = torch.from_numpy(batch["priv_state"]).to(self.device)
        base_action = torch.from_numpy(batch["base_action"]).to(self.device)
        delta_action = torch.from_numpy(batch["delta_action"]).to(self.device)
        reward = torch.from_numpy(batch["reward"]).to(self.device)
        next_feature = torch.from_numpy(batch["next_feature"]).to(self.device)
        next_priv_state = torch.from_numpy(batch["next_priv_state"]).to(self.device)
        next_base_action = torch.from_numpy(batch["next_base_action"]).to(self.device)
        done = torch.from_numpy(batch["done"]).to(self.device)
        with torch.no_grad():
            noise = torch.randn_like(delta_action) * POLICY_NOISE
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            limit_vec = self.actor.limit_vec.to(self.device)
            next_delta = torch.clamp(self.actor_target(next_feature, next_base_action) + noise, -limit_vec, limit_vec)
            q1_t = self.critic1_target(next_feature, next_priv_state, next_base_action, next_delta)
            q2_t = self.critic2_target(next_feature, next_priv_state, next_base_action, next_delta)
            target = reward + (1.0 - done) * DISCOUNT * torch.min(q1_t, q2_t)
        q1 = self.critic1(feature, priv_state, base_action, delta_action)
        q2 = self.critic2(feature, priv_state, base_action, delta_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=5.0)
        self.critic_opt.step()
        logs = {"critic_loss": float(critic_loss.item()), "q1": float(q1.mean().item())}
        self.total_updates += 1
        if self.total_updates % POLICY_DELAY == 0:
            delta = self.actor(feature, base_action)
            actor_loss = -self.critic1(feature, priv_state, base_action, delta).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_opt.step()
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            logs["actor_loss"] = float(actor_loss.item())
        return logs

    def soft_update(self, src, tgt):
        with torch.no_grad():
            for a, b in zip(src.parameters(), tgt.parameters()):
                b.data.mul_(1.0 - TAU).add_(TAU * a.data)

    def save(self, path):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "feature_dim": self.feature_dim,
            "priv_dim": self.priv_dim,
            "action_dim": self.action_dim,
            "arm_dim": self.arm_dim,
            "delta_arm_limit": DELTA_ARM_LIMIT,
            "grip_delta_limit": GRIP_DELTA_LIMIT,
        }, path)


class ReplayBuffer(object):
    def __init__(self, capacity, feature_dim, priv_dim, action_dim):
        self.capacity = int(capacity)
        self.feature = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.priv_state = np.zeros((capacity, priv_dim), dtype=np.float32)
        self.base_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.delta_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_feature = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.next_priv_state = np.zeros((capacity, priv_dim), dtype=np.float32)
        self.next_base_action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, feature, priv_state, base_action, delta_action, reward, next_feature, next_priv_state, next_base_action, done):
        i = self.ptr
        self.feature[i] = feature
        self.priv_state[i] = priv_state
        self.base_action[i] = base_action
        self.delta_action[i] = delta_action
        self.reward[i, 0] = float(reward)
        self.next_feature[i] = next_feature
        self.next_priv_state[i] = next_priv_state
        self.next_base_action[i] = next_base_action
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "feature": self.feature[idx],
            "priv_state": self.priv_state[idx],
            "base_action": self.base_action[idx],
            "delta_action": self.delta_action[idx],
            "reward": self.reward[idx],
            "next_feature": self.next_feature[idx],
            "next_priv_state": self.next_priv_state[idx],
            "next_base_action": self.next_base_action[idx],
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
            base = base_policy.extract(pi["obs_static"], pi["obs_gripper"], pi["future_static"], pi["task"], pi["goal_type"])
            base_action = to_base_action(base)
            delta = agent.act(base["feature"], base_action, add_noise=False)
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
    base = base_policy.extract(pi["obs_static"], pi["obs_gripper"], pi["future_static"], pi["task"], pi["goal_type"])
    feature_dim = int(base["feature"].shape[0])
    priv_dim = int(pi["priv_state"].shape[0])
    replay = ReplayBuffer(BUFFER_SIZE, feature_dim, priv_dim, ACTION_DIM)
    agent = TD3FineTuningAgent(feature_dim, priv_dim, ACTION_DIM, ARM_ACTION_DIM, DEVICE)

    history = []
    best_success = -1.0
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    recent_success = deque(maxlen=20)

    for env_step in range(1, TOTAL_ENV_STEPS + 1):
        base = base_policy.extract(pi["obs_static"], pi["obs_gripper"], pi["future_static"], pi["task"], pi["goal_type"])
        base_action = to_base_action(base)
        if env_step <= RANDOM_WARMUP_STEPS:
            limit_vec = np.asarray([DELTA_ARM_LIMIT] * ARM_ACTION_DIM + [GRIP_DELTA_LIMIT], dtype=np.float32)
            delta = np.random.uniform(low=-limit_vec, high=limit_vec).astype(np.float32)
        else:
            delta = agent.act(base["feature"], base_action, add_noise=True)
        full_action = train_env._compose_full_action(base_action, delta)
        next_pi, reward, done, truncated, step_info = train_env.step(full_action, delta)
        terminal = bool(done or truncated)

        if next_pi is None:
            next_feature = base["feature"].copy()
            next_base_action = base_action.copy()
            next_priv = pi["priv_state"].copy()
        else:
            next_base = base_policy.extract(next_pi["obs_static"], next_pi["obs_gripper"], next_pi["future_static"], next_pi["task"], next_pi["goal_type"])
            next_feature = next_base["feature"].copy()
            next_base_action = to_base_action(next_base)
            next_priv = next_pi["priv_state"].copy()

        replay.add(base["feature"], pi["priv_state"], base_action, delta, reward, next_feature, next_priv, next_base_action, float(terminal))
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
            episode_reward = 0.0
            episode_steps = 0
        else:
            pi = next_pi

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
