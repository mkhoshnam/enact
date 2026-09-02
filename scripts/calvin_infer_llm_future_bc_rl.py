import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

import argparse
import csv
import json
import os

USE_EGL = os.environ.get("CALVIN_USE_EGL", "1") == "1"

if USE_EGL:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "true")

import random
import re
import yaml
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

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import openai
except Exception:
    openai = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default=None):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
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
FINE_TUNING_ACTOR_PATH = env_path(
    "CALVIN_FINE_TUNING_ACTOR_PATH",
    RESULTS_ROOT / "rafc_rl_runs" / RUN_NAME / "policy_best.pt",
)
FUTURE_VIDEO_ROOT = env_path("CALVIN_GENERATED_FUTURE_ROOT", OUT_BASE / "generated_inpainted_calvin_futures")
ONTOLOGY_PATH = env_path("CALVIN_ONTOLOGY_PATH", REPO_ROOT / "ontology" / "calvin_task_ontology.json")
OUTPUT_DIR = env_path("CALVIN_INFERENCE_OUTPUT_DIR", OUT_BASE / "llm_future_policy_runs")
FUTURE_SHIFTS = (-2, 0, 2)
RAFC_INIT_ALPHA = 0.90
RAFC_CENTER_LOGIT_BIAS = 3.0
EVAL_MODE = os.environ.get("CALVIN_EVAL_MODE", "single").strip().lower()
USE_RAFC_EVAL = os.environ.get("CALVIN_USE_RAFC", "0") != "0"
FUTURE_SOURCE = os.environ.get("CALVIN_FUTURE_SOURCE", "generated").strip().lower()
FUTURE_EVAL_MODE = os.environ.get("CALVIN_FUTURE_MODE", "gen").lower()
FUTURE_EVAL_SHIFT = int(os.environ.get("CALVIN_FUTURE_SHIFT", "0"))
WRONG_FUTURE_VIDEO_PATH = os.environ.get("CALVIN_WRONG_FUTURE_VIDEO_PATH", "")
ALLOW_GT_ORACLE = os.environ.get("CALVIN_ALLOW_GT_ORACLE", "0") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_GUI = True
SAVE_VIDEO = True
VIDEO_FPS = 12
IMAGE_SIZE = None
NUM_EPISODES = int(os.environ.get("CALVIN_NUM_EVAL_EPISODES", "20"))
MAX_EPISODE_STEPS = 200
ARM_ACTION_DIM = 6
ACTION_DIM = ARM_ACTION_DIM + 1
TRAIN_SPLIT = 0.90

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


def output_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_future_mode(value):
    mode = str(value).strip().lower()
    aliases = {
        "null": "nofuture",
        "none": "nofuture",
        "generated": "gen",
        "demo": "gt",
        "groundtruth": "gt",
        "ground_truth": "gt",
        "temporal_shift": "shift",
        "temporalshift": "shift",
    }
    return aliases.get(mode, mode)


def default_run_name(seed, future_mode, use_rafc, future_shift=0):
    mode = str(future_mode)
    if mode == "shift":
        shift_tag = "p{}".format(int(future_shift)) if int(future_shift) >= 0 else "m{}".format(abs(int(future_shift)))
        mode = "shift_{}".format(shift_tag)
    policy_tag = "rafc" if use_rafc else "single"
    return "multitask_{}_{}_td3bc_seed{}".format(mode, policy_tag, int(seed))


def parse_args():
    parser = argparse.ArgumentParser(description="CALVIN paper evaluation")
    parser.add_argument("--eval_mode", choices=["single", "table2", "fig4_table3", "table4"], default=os.environ.get("CALVIN_EVAL_MODE", "single").strip().lower())
    parser.add_argument("--future_mode", choices=["nofuture", "gt", "gen", "shift"], default=normalize_future_mode(os.environ.get("CALVIN_FUTURE_MODE", "gen")))
    parser.add_argument("--use_rafc", type=int, choices=[0, 1], default=int(os.environ.get("CALVIN_USE_RAFC", "0")))
    parser.add_argument("--future_shift", type=int, choices=[-6, -4, -2, 0, 2, 4, 6], default=int(os.environ.get("CALVIN_FUTURE_SHIFT", "0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CALVIN_SEED", "42")))
    parser.add_argument("--max_episode_steps", type=int, default=int(os.environ.get("CALVIN_MAX_EPISODE_STEPS", "200")))
    parser.add_argument("--task", default=os.environ.get("CALVIN_EVAL_TASK", ""))
    parser.add_argument("--command", default=os.environ.get("CALVIN_EVAL_COMMAND", ""))
    parser.add_argument("--checkpoint_step", type=int, default=int(os.environ.get("CALVIN_CHECKPOINT_STEP", "-1")))
    parser.add_argument("--checkpoint_path", default=os.environ.get("CALVIN_FINE_TUNING_ACTOR_PATH", ""))
    parser.add_argument("--num_episodes", type=int, default=int(os.environ.get("CALVIN_NUM_EVAL_EPISODES", str(NUM_EPISODES))))
    parser.add_argument("--out_csv", default=os.environ.get("CALVIN_EVAL_OUT_CSV", "results/single_eval.csv"))
    parser.add_argument("--save_video", type=int, choices=[0, 1], default=int(os.environ.get("CALVIN_SAVE_VIDEO", "0")))
    return parser.parse_args()


def configure_from_args(args):
    global EVAL_MODE, USE_RAFC_EVAL, FUTURE_EVAL_MODE, FUTURE_EVAL_SHIFT
    global FUTURE_SOURCE, SEED, MAX_EPISODE_STEPS, NUM_EPISODES, SAVE_VIDEO

    EVAL_MODE = str(args.eval_mode).strip().lower()
    USE_RAFC_EVAL = bool(int(args.use_rafc))
    FUTURE_EVAL_MODE = normalize_future_mode(args.future_mode)
    FUTURE_EVAL_SHIFT = int(args.future_shift)
    FUTURE_SOURCE = future_source_for_mode(FUTURE_EVAL_MODE)
    SEED = int(args.seed)
    MAX_EPISODE_STEPS = int(args.max_episode_steps)
    NUM_EPISODES = int(args.num_episodes)
    SAVE_VIDEO = bool(int(args.save_video))


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
        cfg.env.use_egl = bool(USE_EGL) and not bool(show_gui)
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
    out = {}
    for pth in files:
        m = _EP_RE.search(pth.name)
        if m is not None:
            out[int(m.group(1))] = pth
    if len(out) == 0:
        raise RuntimeError("No episode files found in {}".format(data_root))
    return out


class EpisodeCache(object):
    def __init__(self, data_root, max_items=2048):
        self.episode_file_map = build_episode_file_map(data_root)
        self.cache = OrderedDict()
        self.max_items = int(max_items)

    def get(self, idx):
        idx = int(idx)
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx not in self.episode_file_map:
            raise KeyError("Episode index {} not found".format(idx))
        data = np.load(str(self.episode_file_map[idx]), allow_pickle=True)
        item = {k: data[k] for k in data.files}
        self.cache[idx] = item
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return item


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
    return [segments[int(i)] for i in ids[:n_train]], [segments[int(i)] for i in ids[n_train:]] if len(ids) > 1 else [segments[0]]


def get_u8(obs, key):
    if isinstance(obs, dict) and key in obs:
        return np.asarray(obs[key], dtype=np.uint8)
    if isinstance(obs, dict) and "rgb_obs" in obs and key in obs["rgb_obs"]:
        return np.asarray(obs["rgb_obs"][key], dtype=np.uint8)
    raise KeyError("Could not find '{}' in observation".format(key))


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


def default_future_video_path(task):
    return FUTURE_VIDEO_ROOT / task / "inpainted_robot_future.mp4"


def resolve_future_video_path(task, path_value):
    raw = str(path_value or "").strip()
    if not raw:
        return default_future_video_path(task)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) > 0 and path.parts[0] == task:
        return FUTURE_VIDEO_ROOT / path
    return OUT_BASE / path


def read_future_video(path, count, image_size=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("generated future video not found: {}".format(path))
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
        raise RuntimeError("empty future video: {}".format(path))
    ids = np.linspace(0, len(frames) - 1, num=count)
    ids = np.rint(ids).astype(np.int32)
    return np.stack([frames[int(i)] for i in ids], axis=0)


def make_null_future(future_static):
    future_static = np.asarray(future_static, dtype=np.uint8)
    return np.repeat(future_static[:1], repeats=future_static.shape[0], axis=0).astype(np.uint8)


def make_null_future_clip(current_obs_static, count):
    frame = np.asarray(current_obs_static, dtype=np.uint8)
    if frame.ndim == 4:
        frame = frame[-1]
    return np.repeat(frame[None, ...], repeats=int(count), axis=0).astype(np.uint8)


def shift_future(future_static, shift):
    future_static = np.asarray(future_static, dtype=np.uint8)
    t = int(future_static.shape[0])
    idx = np.arange(t, dtype=np.int32) + int(shift)
    idx = np.clip(idx, 0, t - 1)
    return future_static[idx].astype(np.uint8)


def make_rafc_future_batch(future_static, future_shifts=FUTURE_SHIFTS):
    futures = [make_null_future(future_static)]
    futures.extend(shift_future(future_static, shift) for shift in future_shifts)
    return np.stack(futures, axis=0).astype(np.uint8)


def future_source_for_mode(future_mode):
    if future_mode == "gt":
        return "demo"
    if future_mode == "nofuture":
        return "null"
    return "generated"


def apply_future_eval_mode(future_static, future_mode=None, future_shift=None):
    mode = normalize_future_mode(FUTURE_EVAL_MODE if future_mode is None else future_mode)
    shift = FUTURE_EVAL_SHIFT if future_shift is None else int(future_shift)
    if mode == "nofuture":
        return make_null_future(future_static)
    if mode == "shift" or shift != 0:
        return shift_future(future_static, shift)
    return np.asarray(future_static, dtype=np.uint8)


def write_video(frames, path, fps):
    if imageio is None or len(frames) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)


def short_state(obs, info=None):
    state = {
        "robot_obs": np.asarray(obs.get("robot_obs", []), dtype=np.float32).round(4).tolist() if isinstance(obs, dict) else [],
        "scene_obs": np.asarray(obs.get("scene_obs", []), dtype=np.float32).round(4).tolist() if isinstance(obs, dict) else [],
    }
    if info is not None:
        state["env_info_keys"] = list(info.keys()) if isinstance(info, dict) else []
    return state



def load_ontology(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("ontology json not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    return ontology


def ontology_tasks(ontology):
    raw = ontology.get("tasks", ontology.get("calvin_tasks", {}))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("task_key") or item.get("calvin_task") or item.get("name")
            if key:
                out[str(key)] = item
        return out
    return {}


def task_aliases(task_key, meta):
    aliases = [task_key.replace("_", " ")]
    for key in ["aliases", "language", "commands", "instructions", "related_commands"]:
        vals = meta.get(key, []) if isinstance(meta, dict) else []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            v = str(v).strip()
            if v:
                aliases.append(v)
    return aliases


def pick_task_from_ontology(command, tasks, ontology):
    q = command.lower().replace("-", " ").replace("_", " ")
    task_meta = ontology_tasks(ontology)
    allowed = set(tasks)
    best_task = None
    best_score = -1

    for task, meta in task_meta.items():
        if task not in allowed:
            continue
        score = 0
        for alias in task_aliases(task, meta):
            a = alias.lower().replace("-", " ").replace("_", " ")
            if a and a in q:
                score = max(score, len(a.split()))
        for word in str(meta.get("object", "")).lower().split():
            if word and word in q:
                score += 1
        for word in str(meta.get("motion_type", "")).lower().split():
            if word and word in q:
                score += 1
        if score > best_score:
            best_score = score
            best_task = task

    if best_task is not None:
        return best_task

    ontology_allowed = [t for t in task_meta.keys() if t in allowed]
    if ontology_allowed:
        return ontology_allowed[0]
    return tasks[0]


def fallback_plan(command, tasks, ontology):
    task = pick_task_from_ontology(command, tasks, ontology)
    meta = ontology_tasks(ontology).get(task, {})
    future_video_path = resolve_future_video_path(
        task,
        meta.get("future_video_path", meta.get("generated_video_path", "")),
    )
    return {
        "task_key": task,
        "selected_object": meta.get("object", meta.get("target_object", "")),
        "interaction_part": meta.get("interaction_part", meta.get("part", "")),
        "motion_type": meta.get("motion_type", meta.get("motion", "")),
        "future_caption": meta.get("future_caption", "a robot arm completing the task: {}".format(command)),
        "generated_video_path": str(future_video_path),
        "success_criteria": meta.get("success_criteria", "CALVIN oracle reports task success"),
        "reasoning": "matched the command to the ontology entry",
    }


def query_llm_for_task(command, tasks, obs_state, ontology):
    fallback = fallback_plan(command, tasks, ontology)
    task_meta = ontology_tasks(ontology)

    if not OPENAI_API_KEY or openai is None:
        return fallback

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    client = openai.OpenAI()

    ontology_view = {
        "name": ontology.get("name", "calvin_ontology"),
        "benchmark": ontology.get("benchmark", "CALVIN"),
        "tasks": {t: task_meta.get(t, {}) for t in tasks if t in task_meta},
    }

    system_prompt = """
Map the CALVIN command to one ontology task.
Use only the provided ontology.
Return valid JSON only.
""".strip()

    user_prompt = {
        "command": command,
        "ontology": ontology_view,
        "state": obs_state,
        "return_fields": [
            "task_key",
            "selected_object",
            "interaction_part",
            "motion_type",
            "future_caption",
            "generated_video_path",
            "success_criteria",
            "reasoning",
        ],
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, indent=2)},
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        plan = json.loads(text)
        if plan.get("task_key") not in tasks:
            plan["task_key"] = fallback["task_key"]

        meta = task_meta.get(plan["task_key"], {})
        plan.setdefault("selected_object", meta.get("object", meta.get("target_object", "")))
        plan.setdefault("interaction_part", meta.get("interaction_part", meta.get("part", "")))
        plan.setdefault("motion_type", meta.get("motion_type", meta.get("motion", "")))
        plan.setdefault("future_caption", meta.get("future_caption", fallback["future_caption"]))
        plan.setdefault("success_criteria", meta.get("success_criteria", fallback["success_criteria"]))
        plan["generated_video_path"] = str(resolve_future_video_path(
            plan["task_key"],
            plan.get("generated_video_path", meta.get("future_video_path", fallback["generated_video_path"])),
        ))
        return plan
    except Exception as exc:
        fallback["reasoning"] = "matched the command to the ontology entry"
        return fallback


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
    def __init__(self, arm_dim, chunk_horizon, obs_horizon, future_horizon, num_tasks, hidden_dim=384, num_layers=4, num_heads=8, dropout=0.10, ff_mult=4, pretrained_backbone=True, future_bins=4):
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.obs_horizon = int(obs_horizon)
        self.future_horizon = int(future_horizon)
        self.future_bins = int(future_bins)
        self.num_tasks = int(num_tasks)
        self.image_encoder = ResNet18Encoder(pretrained=pretrained_backbone)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, hidden_dim)
        self.task_emb = nn.Embedding(num_tasks, hidden_dim)
        self.goal_emb = nn.Embedding(4, hidden_dim)
        self.future_pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.future_bins, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
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
        return self.image_proj(self.image_encoder(x)).view(b, t, -1)

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
        future_tokens = x[:, future_start:future_start + self.future_horizon]
        future_bins = F.adaptive_avg_pool1d(
            future_tokens.transpose(1, 2),
            self.future_bins,
        ).transpose(1, 2)
        future_latent = self.future_pool_proj(
            future_bins.reshape(bsz, -1)
        )
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
        self.future_bins = int(cfg.get("future_bins", 4))
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
            future_bins=self.future_bins,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device).eval()
        for pth in self.model.parameters():
            pth.requires_grad = False

    def extract_many(self, obs_static, obs_gripper, future_static_batch, task_name, goal_type=3):
        future_static_batch = np.asarray(future_static_batch, dtype=np.uint8)
        if future_static_batch.ndim == 4:
            future_static_batch = future_static_batch[None, ...]
        batch_size = int(future_static_batch.shape[0])
        task_id = int(self.task_to_id[task_name])
        obs_static_batch = np.repeat(np.asarray(obs_static, dtype=np.uint8)[None, ...], batch_size, axis=0)
        obs_gripper_batch = np.repeat(np.asarray(obs_gripper, dtype=np.uint8)[None, ...], batch_size, axis=0)
        obs_static_t = torch.from_numpy(obs_static_batch).to(self.device)
        obs_gripper_t = torch.from_numpy(obs_gripper_batch).to(self.device)
        future_static_t = torch.from_numpy(future_static_batch).to(self.device)
        task_t = torch.full((batch_size,), task_id, dtype=torch.long, device=self.device)
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
        self.g_dim = int(g_dim)
        self.register_buffer("limit_vec", torch.tensor([arm_limit] * arm_dim + [grip_limit], dtype=torch.float32).view(1, -1))
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim) + self.g_dim + int(action_dim), 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

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


class FineTuningActorWrapper(object):
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.feature_dim = int(ckpt["feature_dim"])
        self.g_dim = int(ckpt.get("g_dim", 0))
        self.action_dim = int(ckpt.get("action_dim", ACTION_DIM))
        self.arm_dim = int(ckpt.get("arm_dim", ARM_ACTION_DIM))
        self.arm_limit = float(ckpt.get("delta_arm_limit", 0.10))
        self.grip_limit = float(ckpt.get("grip_delta_limit", 2.0))
        self.actor = FineTuningActor(self.feature_dim, self.g_dim, self.action_dim, self.arm_dim, self.arm_limit, self.grip_limit).to(device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()
        self.uses_rafc = bool(ckpt.get("uses_rafc", "gate_state_dict" in ckpt))
        self.future_shifts = tuple(ckpt.get("future_shifts", FUTURE_SHIFTS))
        self.gate = None
        if self.uses_rafc:
            num_shifts = int(ckpt.get("num_future_shifts", len(self.future_shifts)))
            self.gate = FutureGate(self.feature_dim, num_shifts).to(device)
            self.gate.load_state_dict(ckpt["gate_state_dict"])
            self.gate.eval()
        self.limit_vec_np = np.asarray([self.arm_limit] * self.arm_dim + [self.grip_limit], dtype=np.float32)

    def act(self, feature_np, g_np, base_action_np):
        feature = torch.from_numpy(feature_np).unsqueeze(0).to(self.device)
        if self.g_dim > 0:
            g_t = torch.from_numpy(g_np).unsqueeze(0).to(self.device)
        else:
            g_t = torch.zeros((1, 0), dtype=feature.dtype, device=self.device)
        base_action = torch.from_numpy(base_action_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.actor(feature, g_t, base_action)[0].cpu().numpy()
        return np.clip(delta, -self.limit_vec_np, self.limit_vec_np).astype(np.float32)

    def act_rafc(self, rafc):
        if self.gate is None:
            raise RuntimeError("RAFC checkpoint expected gate_state_dict, but no gate was loaded")
        feature_null = torch.from_numpy(rafc["feature_null"]).unsqueeze(0).to(self.device)
        feature_candidates = torch.from_numpy(rafc["feature_candidates"]).unsqueeze(0).to(self.device)
        g_null = torch.from_numpy(rafc["g_null"]).unsqueeze(0).to(self.device)
        g_candidates = torch.from_numpy(rafc["g_candidates"]).unsqueeze(0).to(self.device)
        base_action_null = torch.from_numpy(rafc["base_action_null"]).unsqueeze(0).to(self.device)
        base_action_candidates = torch.from_numpy(rafc["base_action_candidates"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature, alpha, weights = self.gate(feature_null, feature_candidates)
            g_t = mix_with_gate(alpha, weights, g_null, g_candidates)
            base_action = mix_with_gate(alpha, weights, base_action_null, base_action_candidates)
            delta = self.actor(feature, g_t, base_action)[0].cpu().numpy()
        return np.clip(delta, -self.limit_vec_np, self.limit_vec_np).astype(np.float32), {
            "base_action": base_action[0].cpu().numpy().astype(np.float32),
            "alpha": alpha[0].cpu().numpy().astype(np.float32),
            "weights": weights[0].cpu().numpy().astype(np.float32),
        }


class LLMFuturePolicyRunner(object):
    def __init__(self, segments, base_policy, fine_tuning_actor, cache, plan, future_mode=None, future_shift=None, show_gui=None):
        self.segments = list(segments)
        self.base_policy = base_policy
        self.fine_tuning_actor = fine_tuning_actor
        self.cache = cache
        self.plan = plan
        self.task = plan["task_key"]
        self.future_mode = normalize_future_mode(FUTURE_EVAL_MODE if future_mode is None else future_mode)
        if self.future_mode == "gt" and not ALLOW_GT_ORACLE:
            raise ValueError(
                "GTFuture is an oracle condition. Run "
                "scripts/calvin_infer_gt_future_oracle.py instead."
            )
        self.future_shift = FUTURE_EVAL_SHIFT if future_shift is None else int(future_shift)
        self.future_source = future_source_for_mode(self.future_mode)
        self.show_gui = SHOW_GUI if show_gui is None else bool(show_gui)
        self.env = None
        self.tasks_oracle = None
        self.current_segment = None
        self.start_info = None
        self.obs_static_hist = deque(maxlen=base_policy.obs_horizon)
        self.obs_gripper_hist = deque(maxlen=base_policy.obs_horizon)
        self.last_obs = None
        future_path = WRONG_FUTURE_VIDEO_PATH if self.future_mode == "wrong" and WRONG_FUTURE_VIDEO_PATH else plan["generated_video_path"]
        self.future_path = resolve_future_video_path(self.task, future_path)
        self.generated_future_static = None
        if self.future_source == "generated" and self.future_path.exists():
            self.generated_future_static = apply_future_eval_mode(
                read_future_video(self.future_path, base_policy.future_horizon, IMAGE_SIZE),
                self.future_mode,
                self.future_shift,
            )
        self.future_static = None

    def _ensure_env(self):
        if self.env is None:
            self.env, self.tasks_oracle = make_env(self.show_gui)

    def _segment_for_task(self, episode_id):
        candidates = [s for s in self.segments if s["task"] == self.task]
        if len(candidates) == 0:
            raise RuntimeError("No eval segments found for task {}".format(self.task))
        return candidates[episode_id % len(candidates)]

    def _demo_future(self, aligned_idx):
        seg_end = int(self.current_segment["global_end_idx"])
        fut_idx = sample_future_indices(aligned_idx, seg_end, self.base_policy.future_horizon)
        return np.stack(
            [
                resize_if_needed(np.asarray(self.cache.get(int(j))["rgb_static"], dtype=np.uint8), IMAGE_SIZE)
                for j in fut_idx
            ],
            axis=0,
        )

    def _select_future(self, start_idx, init_static):
        if self.future_source == "generated":
            if self.generated_future_static is not None:
                return self.generated_future_static
            raise FileNotFoundError(
                "Generated future is missing; visual-only inference "
                "forbids demonstration fallback: {}".format(
                    self.future_path
                )
            )
        if self.future_source == "demo":
            if not ALLOW_GT_ORACLE:
                raise ValueError(
                    "Demonstration futures are restricted to the explicitly "
                    "labelled GTFuture oracle runner"
                )
            return apply_future_eval_mode(self._demo_future(start_idx), self.future_mode, self.future_shift)
        if self.future_source == "null":
            return make_null_future_clip(init_static, self.base_policy.future_horizon)
        raise ValueError("Unknown future_source: {}".format(self.future_source))

    def reset(self, episode_id):
        self._ensure_env()
        self.current_segment = self._segment_for_task(episode_id)
        start_idx = int(self.current_segment["global_start_idx"])
        item = self.cache.get(start_idx)
        obs = self.env.reset(robot_obs=np.asarray(item["robot_obs"], dtype=np.float32), scene_obs=np.asarray(item["scene_obs"], dtype=np.float32))
        self.start_info = self.env.get_info()
        self.last_obs = obs
        self.obs_static_hist.clear()
        self.obs_gripper_hist.clear()
        init_static = resize_if_needed(get_u8(obs, "rgb_static"), IMAGE_SIZE)
        init_gripper = resize_if_needed(get_u8(obs, "rgb_gripper"), IMAGE_SIZE)
        self.future_static = self._select_future(start_idx, init_static)
        for _ in range(self.base_policy.obs_horizon):
            self.obs_static_hist.append(init_static.copy())
            self.obs_gripper_hist.append(init_gripper.copy())
        return self._policy_input(), {"task": self.task, "segment_id": int(self.current_segment["segment_id"])}

    def _policy_input(self):
        return {
            "obs_static": np.stack(list(self.obs_static_hist), axis=0).astype(np.uint8),
            "obs_gripper": np.stack(list(self.obs_gripper_hist), axis=0).astype(np.uint8),
            "future_static": self.future_static.astype(np.uint8),
            "task": self.task,
            "goal_type": 3,
        }

    def compose_action(self, base_action, delta_action):
        arm = np.clip(base_action[:ARM_ACTION_DIM] + delta_action[:ARM_ACTION_DIM], -1.0, 1.0).astype(np.float32)
        grip_score = float(base_action[ARM_ACTION_DIM] + delta_action[ARM_ACTION_DIM])
        grip = 1.0 if grip_score >= 0.0 else -1.0
        return np.concatenate([arm, np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32)

    def step(self, action):
        step_res = self.env.step(np.asarray(action, dtype=np.float32))
        if len(step_res) == 5:
            obs, raw_reward, terminated_raw, truncated_raw, _ = step_res
            env_done = bool(terminated_raw) or bool(truncated_raw)
        else:
            obs, raw_reward, env_done, _ = step_res
        self.last_obs = obs
        self.obs_static_hist.append(resize_if_needed(get_u8(obs, "rgb_static"), IMAGE_SIZE))
        self.obs_gripper_hist.append(resize_if_needed(get_u8(obs, "rgb_gripper"), IMAGE_SIZE))
        curr_info = self.env.get_info()
        success = oracle_success(self.tasks_oracle, self.start_info, curr_info, self.task)
        try:
            reward_value = float(raw_reward)
        except Exception:
            reward_value = 0.0
        return self._policy_input(), bool(success), bool(env_done), reward_value, {"success": bool(success), "task": self.task, "segment_id": int(self.current_segment["segment_id"])}

    def frame(self):
        if self.last_obs is None:
            return None
        return resize_if_needed(get_u8(self.last_obs, "rgb_static"), IMAGE_SIZE).copy()

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass
        self.env = None


def to_base_action(base):
    return np.concatenate([base["base_arm"][:ARM_ACTION_DIM], np.asarray([base["base_grip"]], dtype=np.float32)], axis=0).astype(np.float32)


def extract_rafc_inputs(base_policy, pi, future_shifts=FUTURE_SHIFTS):
    future_batch = make_rafc_future_batch(pi["future_static"], future_shifts=future_shifts)
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


SINGLE_COLUMNS = [
    "task",
    "future_mode",
    "use_rafc",
    "future_shift",
    "seed",
    "checkpoint_step",
    "max_episode_steps",
    "success_rate",
    "mean_return",
    "alpha",
    "w_minus2",
    "w_0",
    "w_plus2",
]

TABLE_COLUMNS = ["condition"] + SINGLE_COLUMNS
TABLE4_COLUMNS = ["condition", "future_mode", "future_shift", "seed", "task", "alpha", "w_minus2", "w_0", "w_plus2"]


def write_csv_rows(path, fieldnames, rows):
    path = output_path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def plan_for_task(task, ontology):
    task_meta = ontology_tasks(ontology).get(task, {})
    future_path = resolve_future_video_path(
        task,
        task_meta.get("future_video_path", task_meta.get("generated_video_path", "")),
    )
    return {
        "task_key": task,
        "selected_object": task_meta.get("object", task_meta.get("target_object", "")),
        "interaction_part": task_meta.get("interaction_part", task_meta.get("part", "")),
        "motion_type": task_meta.get("motion_type", task_meta.get("motion", "")),
        "future_caption": task_meta.get("future_caption", "a robot arm completing the task {}".format(task)),
        "generated_video_path": str(future_path),
        "success_criteria": task_meta.get("success_criteria", "CALVIN oracle reports task success"),
        "reasoning": "selected from the evaluation task list",
    }


def checkpoint_filename(checkpoint_step):
    if int(checkpoint_step) > 0:
        return "policy_step_{:06d}.pt".format(int(checkpoint_step))
    return "policy_best.pt"


def resolve_actor_path(seed, future_mode, use_rafc, future_shift, checkpoint_step, checkpoint_path=""):
    if int(checkpoint_step) == 0:
        return None
    if checkpoint_path:
        path = output_path(checkpoint_path)
        if path.exists():
            return path
        raise FileNotFoundError("Checkpoint path does not exist: {}".format(path))

    candidates = []
    modes = [normalize_future_mode(future_mode)]
    if modes[0] == "shift":
        modes.append("gen")
    for mode in modes:
        run_shift = future_shift if mode == "shift" else 0
        run_dir = RESULTS_ROOT / "rafc_rl_runs" / default_run_name(seed, mode, use_rafc, run_shift)
        if int(checkpoint_step) > 0:
            candidates.append(run_dir / checkpoint_filename(checkpoint_step))
            if int(checkpoint_step) == 140000:
                candidates.append(run_dir / "policy_final.pt")
        else:
            candidates.append(run_dir / "policy_best.pt")
            candidates.append(run_dir / "policy_final.pt")

    old_run_dir = RESULTS_ROOT / "rafc_rl_runs" / "multitask_step6_rafc_td3bc_seed{}".format(int(seed))
    if int(checkpoint_step) > 0:
        candidates.append(old_run_dir / checkpoint_filename(checkpoint_step))
    else:
        candidates.append(old_run_dir / "policy_best.pt")
        candidates.append(Path(FINE_TUNING_ACTOR_PATH))

    for path in candidates:
        if Path(path).exists():
            return Path(path)
    raise FileNotFoundError("Could not find RL checkpoint. Tried: {}".format(", ".join(str(p) for p in candidates)))


def load_actor_for_config(actor_cache, seed, future_mode, use_rafc, future_shift, checkpoint_step, checkpoint_path=""):
    actor_path = resolve_actor_path(seed, future_mode, use_rafc, future_shift, checkpoint_step, checkpoint_path)
    if actor_path is None:
        return None, "bc"
    key = str(actor_path)
    if key not in actor_cache:
        actor_cache[key] = FineTuningActorWrapper(actor_path, DEVICE)
    return actor_cache[key], str(actor_path)


def run_task_episodes(eval_segments, base_policy, fine_tuning_actor, cache, plan, future_mode, future_shift, use_rafc, num_episodes, max_episode_steps, save_video=False):
    runner = LLMFuturePolicyRunner(
        eval_segments,
        base_policy,
        fine_tuning_actor,
        cache,
        plan,
        future_mode=future_mode,
        future_shift=future_shift,
        show_gui=False,
    )
    results = []
    gate_alphas = []
    gate_weights = []
    try:
        for ep in range(int(num_episodes)):
            pi, info = runner.reset(ep)
            frames = []
            fr = runner.frame()
            if fr is not None:
                frames.append(fr)
            success = False
            env_done = False
            steps = 0
            episode_return = 0.0
            last_info = info
            while not success and not env_done and steps < int(max_episode_steps):
                if fine_tuning_actor is None:
                    base = base_policy.extract(pi["obs_static"], pi["obs_gripper"], pi["future_static"], pi["task"], pi["goal_type"])
                    base_action = to_base_action(base)
                    delta = np.zeros((ACTION_DIM,), dtype=np.float32)
                elif use_rafc:
                    if fine_tuning_actor.gate is None:
                        raise RuntimeError("Requested RAFC evaluation, but the checkpoint does not contain a gate")
                    rafc = extract_rafc_inputs(base_policy, pi, fine_tuning_actor.future_shifts)
                    delta, gate_info = fine_tuning_actor.act_rafc(rafc)
                    base_action = gate_info["base_action"]
                    gate_alphas.append(float(gate_info["alpha"].mean()))
                    gate_weights.append(np.asarray(gate_info["weights"], dtype=np.float32))
                else:
                    base = base_policy.extract(pi["obs_static"], pi["obs_gripper"], pi["future_static"], pi["task"], pi["goal_type"])
                    base_action = to_base_action(base)
                    delta = fine_tuning_actor.act(base["feature"], base["g_t"], base_action)
                action = runner.compose_action(base_action, delta)
                pi, success, env_done, reward, last_info = runner.step(action)
                episode_return += float(reward)
                steps += 1
                fr = runner.frame()
                if fr is not None:
                    frames.append(fr)
            results.append({
                "episode": ep + 1,
                "task": plan["task_key"],
                "success": bool(success),
                "steps": int(steps),
                "return": float(episode_return),
                "segment_id": int(last_info.get("segment_id", -1)),
            })
            if save_video and len(frames) > 0:
                write_video(frames, OUTPUT_DIR / "episode_{:02d}_{}.mp4".format(ep + 1, plan["task_key"]), VIDEO_FPS)
    finally:
        runner.close()
        try:
            p.disconnect()
        except Exception:
            pass

    weights = np.mean(np.stack(gate_weights, axis=0), axis=0) if gate_weights else np.full((len(FUTURE_SHIFTS),), np.nan, dtype=np.float32)
    return {
        "success_rate": float(sum(int(r["success"]) for r in results) / max(len(results), 1)),
        "mean_return": float(np.mean([r["return"] for r in results])) if results else 0.0,
        "alpha": float(np.mean(gate_alphas)) if gate_alphas else "",
        "weights": [float(x) for x in weights] if gate_weights else ["", "", ""],
        "episodes": results,
    }


def evaluate_config(context, actor_cache, task, future_mode, use_rafc, future_shift, seed, checkpoint_step, max_episode_steps, num_episodes, checkpoint_path="", save_video=False):
    set_seed(seed)
    segments = context["segments"]
    _, eval_segments = split_segments(segments, TRAIN_SPLIT, seed)
    plan = plan_for_task(task, context["ontology"])
    fine_tuning_actor, resolved_actor = load_actor_for_config(
        actor_cache,
        seed,
        future_mode,
        bool(use_rafc),
        future_shift,
        checkpoint_step,
        checkpoint_path,
    )
    metrics = run_task_episodes(
        eval_segments,
        context["base_policy"],
        fine_tuning_actor,
        context["cache"],
        plan,
        future_mode,
        future_shift,
        bool(use_rafc) and fine_tuning_actor is not None,
        num_episodes,
        max_episode_steps,
        save_video=save_video,
    )
    weights = metrics["weights"]
    return {
        "task": task,
        "future_mode": normalize_future_mode(future_mode),
        "use_rafc": int(bool(use_rafc) and fine_tuning_actor is not None),
        "future_shift": int(future_shift),
        "seed": int(seed),
        "checkpoint_step": int(checkpoint_step),
        "max_episode_steps": int(max_episode_steps),
        "success_rate": metrics["success_rate"],
        "mean_return": metrics["mean_return"],
        "alpha": metrics["alpha"],
        "w_minus2": weights[0] if len(weights) > 0 else "",
        "w_0": weights[1] if len(weights) > 1 else "",
        "w_plus2": weights[2] if len(weights) > 2 else "",
        "checkpoint_path": resolved_actor,
    }


def write_learning_curve_png(rows, path):
    path = output_path(path)
    ensure_dir(path.parent)
    width, height = 1100, 700
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 90, width - 50, 70, height - 85
    cv2.line(img, (left, bottom), (right, bottom), (0, 0, 0), 2)
    cv2.line(img, (left, bottom), (left, top), (0, 0, 0), 2)
    cv2.putText(img, "CALVIN BC+RL learning curve", (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    checkpoints = sorted({int(r["checkpoint_step"]) for r in rows})
    max_step = max(max(checkpoints), 1)
    colors = {"nofuture": (180, 80, 50), "gt": (50, 140, 60), "gen": (40, 90, 210)}
    for yv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(bottom - yv * (bottom - top))
        cv2.line(img, (left, y), (right, y), (230, 230, 230), 1)
        cv2.putText(img, "{:.0f}%".format(yv * 100.0), (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    for i, step in enumerate(checkpoints):
        x = int(left + (step / float(max_step)) * (right - left))
        cv2.line(img, (x, bottom), (x, bottom + 6), (0, 0, 0), 1)
        cv2.putText(img, str(step), (x - 28, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    for future_mode in ["nofuture", "gt", "gen"]:
        points = []
        for step in checkpoints:
            vals = [float(r["success_rate"]) for r in rows if r["future_mode"] == future_mode and int(r["checkpoint_step"]) == step]
            if vals:
                x = int(left + (step / float(max_step)) * (right - left))
                y = int(bottom - np.clip(np.mean(vals), 0.0, 1.0) * (bottom - top))
                points.append((x, y))
        if len(points) > 1:
            cv2.polylines(img, [np.asarray(points, dtype=np.int32)], False, colors[future_mode], 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(img, point, 5, colors[future_mode], -1, cv2.LINE_AA)
    legend_x = right - 230
    for j, future_mode in enumerate(["nofuture", "gt", "gen"]):
        y = top + 28 * j
        cv2.line(img, (legend_x, y), (legend_x + 35, y), colors[future_mode], 4, cv2.LINE_AA)
        cv2.putText(img, future_mode, (legend_x + 45, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return path


def prepare_context():
    ensure_dir(OUTPUT_DIR)
    if CALVIN_ROOT is None:
        raise RuntimeError("Set CALVIN_ROOT before running CALVIN inference")
    if DATA_ROOT is None:
        raise RuntimeError("Set CALVIN_DATA_ROOT or CALVIN_ROOT before running CALVIN inference")
    if not BC_CKPT_PATH.exists():
        raise FileNotFoundError("Missing BC checkpoint: {}".format(BC_CKPT_PATH))
    os.chdir(CALVIN_ROOT)
    os.environ["CALVIN_ROOT"] = str(CALVIN_ROOT)
    segments, tasks, _ = load_segments(SEGMENTS_JSON)
    return {
        "segments": segments,
        "tasks": list(tasks),
        "ontology": load_ontology(ONTOLOGY_PATH),
        "cache": EpisodeCache(DATA_ROOT),
        "base_policy": FrozenBCPolicy(BC_CKPT_PATH, DEVICE),
    }


def run_single(args, context, actor_cache):
    if args.command:
        plan = fallback_plan(args.command, context["tasks"], context["ontology"])
        task = plan["task_key"]
    else:
        task = args.task.strip() if args.task.strip() else context["tasks"][0]
    row = evaluate_config(
        context,
        actor_cache,
        task,
        args.future_mode,
        bool(args.use_rafc),
        args.future_shift,
        args.seed,
        args.checkpoint_step,
        args.max_episode_steps,
        args.num_episodes,
        checkpoint_path=args.checkpoint_path,
        save_video=bool(args.save_video),
    )
    out_path = write_csv_rows(args.out_csv, SINGLE_COLUMNS, [row])
    print("saved:", out_path)
    print("success_rate={:.3f} mean_return={:.3f}".format(float(row["success_rate"]), float(row["mean_return"])))


def run_table2(args, context, actor_cache):
    rows = []
    if ALLOW_GT_ORACLE:
        settings = [("GTFuture (oracle)", "gt", False)]
        output_file = "results/table2_gt_oracle_temporal_shift.csv"
    else:
        settings = [
            ("GenFuture", "gen", False),
            ("GenFuture+RAFC", "gen", True),
        ]
        output_file = "results/table2_visual_temporal_shift.csv"
    for condition, future_mode, use_rafc in settings:
        for future_shift in [-6, -4, -2, 0, 2, 4, 6]:
            for task in context["tasks"]:
                for seed in [42, 43, 44]:
                    row = evaluate_config(
                        context,
                        actor_cache,
                        task,
                        future_mode,
                        use_rafc,
                        future_shift,
                        seed,
                        args.checkpoint_step,
                        args.max_episode_steps,
                        args.num_episodes,
                    )
                    row["condition"] = condition
                    rows.append(row)
                    print("table2 {} shift={} task={} seed={} success={:.3f}".format(condition, future_shift, task, seed, row["success_rate"]))
    out_path = write_csv_rows(output_file, TABLE_COLUMNS, rows)
    print("saved:", out_path)


def run_fig4_table3(args, context, actor_cache):
    rows = []
    future_modes = ["gt"] if ALLOW_GT_ORACLE else ["nofuture", "gen"]
    for future_mode in future_modes:
        for checkpoint_step in [0, 30000, 60000, 90000, 120000, 140000]:
            for task in context["tasks"]:
                for seed in [42, 43, 44]:
                    row = evaluate_config(
                        context,
                        actor_cache,
                        task,
                        future_mode,
                        False,
                        0,
                        seed,
                        checkpoint_step,
                        args.max_episode_steps,
                        args.num_episodes,
                    )
                    row["condition"] = "Fig4/Table3"
                    rows.append(row)
                    print("fig4 future={} ckpt={} task={} seed={} success={:.3f}".format(future_mode, checkpoint_step, task, seed, row["success_rate"]))
    if ALLOW_GT_ORACLE:
        output_csv = "results/fig4_table3_gt_oracle_learning_curve.csv"
        output_image = "img/ex_rl_gt_oracle.png"
    else:
        output_csv = "results/fig4_table3_visual_learning_curve.csv"
        output_image = "img/ex_rl_visual.png"
    out_path = write_csv_rows(output_csv, TABLE_COLUMNS, rows)
    img_path = write_learning_curve_png(rows, output_image)
    print("saved:", out_path)
    print("saved:", img_path)


def run_table4(args, context, actor_cache):
    rows = []
    for future_shift in [0, -4, 4]:
        condition = "gen_shift_{:+d}".format(future_shift)
        for task in context["tasks"]:
            for seed in [42, 43, 44]:
                row = evaluate_config(
                    context,
                    actor_cache,
                    task,
                    "gen",
                    True,
                    future_shift,
                    seed,
                    args.checkpoint_step,
                    args.max_episode_steps,
                    args.num_episodes,
                )
                row["condition"] = condition
                rows.append(row)
                print("table4 {} task={} seed={} alpha={}".format(condition, task, seed, row["alpha"]))
    out_path = write_csv_rows("results/table4_gate_diagnostics.csv", TABLE4_COLUMNS, rows)
    print("saved:", out_path)


def main():
    args = parse_args()
    configure_from_args(args)
    set_seed(SEED)
    context = prepare_context()
    actor_cache = {}
    if EVAL_MODE == "single":
        run_single(args, context, actor_cache)
    elif EVAL_MODE == "table2":
        run_table2(args, context, actor_cache)
    elif EVAL_MODE == "fig4_table3":
        run_fig4_table3(args, context, actor_cache)
    elif EVAL_MODE == "table4":
        run_table4(args, context, actor_cache)
    else:
        raise ValueError("Unknown eval_mode: {}".format(EVAL_MODE))


if __name__ == "__main__":
    main()
