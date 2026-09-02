import argparse
import csv
import json
import os
import random
import re
import yaml
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import pybullet as p
import torch
from omegaconf import OmegaConf

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from SFP_training import (
    ConditionalUnet1D,
    GripperHead,
    SFPConditionEncoder,
    TASKS,
    normalize_minmax,
    unnormalize_minmax,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default=None):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


USE_EGL = os.environ.get("CALVIN_USE_EGL", "1") == "1"
if USE_EGL:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYGLET_HEADLESS", "true")

CALVIN_ROOT = env_path("CALVIN_ROOT")
DATA_ROOT = env_path(
    "CALVIN_DATA_ROOT",
    CALVIN_ROOT / "dataset/task_D_D/training" if CALVIN_ROOT is not None else None,
)
OUT_BASE = env_path("ENACT_CALVIN_OUT_BASE", REPO_ROOT / "outputs")
SEGMENTS_JSON = env_path("CALVIN_SEGMENTS_JSON", OUT_BASE / "calvin" / "segments_future_bc.json")
SFP_CKPT_PATH = env_path("CALVIN_SFP_CKPT_PATH", OUT_BASE / "calvin_sfp" / "sfp_policy_best.pt")
GENERATED_FUTURE_ROOT = env_path("CALVIN_GENERATED_FUTURE_ROOT", OUT_BASE / "generated_inpainted_calvin_futures")
OUTPUT_DIR = env_path("CALVIN_SFP_INFERENCE_OUTPUT_DIR", OUT_BASE / "sfp_policy_runs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SPLIT = 0.90
IMAGE_SIZE = None
ARM_ACTION_DIM = 6
ACTION_DIM = ARM_ACTION_DIM + 1
MAX_EPISODE_STEPS = 200
NUM_EPISODES = 20
VIDEO_FPS = 12
FUTURE_SHIFTS = (-2, 0, 2)
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
    if len(files) == 0:
        raise RuntimeError("No episode_*.npz files found in {}".format(data_root))
    out = {}
    for pth in files:
        m = _EP_RE.search(pth.name)
        if m is not None:
            out[int(m.group(1))] = pth
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
        tasks = sorted({normalize_text(s["task"]).replace(" ", "_") for s in segments if "task" in s})
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


def shift_future(future_static, shift):
    future_static = np.asarray(future_static, dtype=np.uint8)
    t = int(future_static.shape[0])
    idx = np.arange(t, dtype=np.int32) + int(shift)
    idx = np.clip(idx, 0, t - 1)
    return future_static[idx].astype(np.uint8)


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


def write_video(frames, path, fps):
    if imageio is None or len(frames) == 0:
        return
    ensure_dir(path.parent)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)


class SFPPolicy(object):
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.device = device
        self.cfg = cfg
        self.tasks = list(cfg.get("tasks", TASKS))
        self.task_to_id = {t: i for i, t in enumerate(self.tasks)}
        self.arm_dim = int(cfg["arm_dim"])
        self.action_horizon = int(cfg["action_horizon"])
        self.obs_horizon = int(cfg["obs_horizon"])
        self.future_horizon = int(cfg["future_horizon"])
        self.sigma0 = float(cfg.get("sigma0", 0.0))
        self.flow_gain = float(cfg.get("flow_gain", 10.0))
        self.arm_min = np.asarray(ckpt["arm_action_min"], dtype=np.float32)
        self.arm_max = np.asarray(ckpt["arm_action_max"], dtype=np.float32)
        self.condition_encoder = SFPConditionEncoder(
            obs_horizon=self.obs_horizon,
            future_horizon=self.future_horizon,
            num_tasks=len(self.tasks),
            image_embed_dim=int(cfg.get("image_embed_dim", 64)),
            task_embed_dim=int(cfg.get("task_embed_dim", 32)),
            goal_embed_dim=int(cfg.get("goal_embed_dim", 16)),
            pretrained_backbone=False,
        ).to(device)
        self.velocity_model = ConditionalUnet1D(
            input_dim=self.arm_dim,
            global_cond_dim=self.condition_encoder.out_dim,
            down_dims=tuple(cfg.get("down_dims", [256, 512, 1024])),
            kernel_size=int(cfg.get("kernel_size", 5)),
            n_groups=int(cfg.get("n_groups", 8)),
            sin_embedding_scale=1.0,
        ).to(device)
        self.gripper_head = GripperHead(self.condition_encoder.out_dim + 1).to(device)
        self.condition_encoder.load_state_dict(ckpt["condition_encoder_state_dict"])
        self.velocity_model.load_state_dict(ckpt["velocity_model_state_dict"])
        self.gripper_head.load_state_dict(ckpt["gripper_head_state_dict"])
        self.condition_encoder.eval()
        self.velocity_model.eval()
        self.gripper_head.eval()

    def action_chunk(self, obs_static, obs_gripper, future_static, task, goal_type=3):
        obs_static_t = torch.from_numpy(np.asarray(obs_static, dtype=np.uint8)[None, ...]).to(self.device)
        obs_gripper_t = torch.from_numpy(np.asarray(obs_gripper, dtype=np.uint8)[None, ...]).to(self.device)
        future_static_t = torch.from_numpy(np.asarray(future_static, dtype=np.uint8)[None, ...]).to(self.device)
        task_t = torch.tensor([self.task_to_id[task]], dtype=torch.long, device=self.device)
        goal_t = torch.tensor([int(goal_type)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            gcond = self.condition_encoder(obs_static_t, obs_gripper_t, future_static_t, task_t, goal_t)
            na = torch.zeros((1, 1, self.arm_dim), dtype=torch.float32, device=self.device)
            if self.sigma0 > 0.0:
                na = na + self.sigma0 * torch.randn_like(na)
            actions = []
            dt = 1.0 / float(max(self.action_horizon - 1, 1))
            for i in range(self.action_horizon):
                t_val = float(i) * dt
                t = torch.tensor([t_val], dtype=torch.float32, device=self.device)
                v = self.velocity_model(na, t, gcond)
                na = na + v * dt
                arm_norm = np.clip(na[0, 0].detach().cpu().numpy(), -1.0, 1.0)
                arm = unnormalize_minmax(arm_norm, self.arm_min, self.arm_max)
                logit = self.gripper_head(torch.cat([gcond, t.view(1, 1)], dim=-1))
                grip = 1.0 if float(torch.sigmoid(logit).item()) > 0.5 else -1.0
                actions.append(np.concatenate([arm[:self.arm_dim], np.asarray([grip], dtype=np.float32)], axis=0).astype(np.float32))
        return actions


class SFPFutureRunner(object):
    def __init__(self, segments, policy, cache, task, future_mode="gen", future_shift=0, show_gui=False):
        self.segments = list(segments)
        self.policy = policy
        self.cache = cache
        self.task = task
        self.future_mode = normalize_future_mode(future_mode)
        self.future_shift = int(future_shift)
        self.show_gui = bool(show_gui)
        self.env = None
        self.tasks_oracle = None
        self.current_segment = None
        self.start_info = None
        self.last_obs = None
        self.obs_static_hist = deque(maxlen=policy.obs_horizon)
        self.obs_gripper_hist = deque(maxlen=policy.obs_horizon)
        self.future_static = None

    def _ensure_env(self):
        if self.env is None:
            self.env, self.tasks_oracle = make_env(self.show_gui)

    def _segment_for_task(self, episode_id):
        candidates = [s for s in self.segments if s["task"] == self.task]
        if len(candidates) == 0:
            raise RuntimeError("No eval segments found for task {}".format(self.task))
        return candidates[episode_id % len(candidates)]

    def _demo_future(self, start_idx):
        seg_end = int(self.current_segment["global_end_idx"])
        fut_idx = sample_future_indices(start_idx, seg_end, self.policy.future_horizon)
        return np.stack([resize_if_needed(np.asarray(self.cache.get(int(j))["rgb_static"], dtype=np.uint8), IMAGE_SIZE) for j in fut_idx], axis=0)

    def _generated_future(self):
        path = GENERATED_FUTURE_ROOT / self.task / "inpainted_robot_future.mp4"
        if path.exists():
            return read_future_video(path, self.policy.future_horizon, IMAGE_SIZE)
        return None

    def _select_future(self, start_idx, init_static):
        if self.future_mode == "nofuture":
            return np.repeat(init_static[None, ...], repeats=self.policy.future_horizon, axis=0).astype(np.uint8)
        if self.future_mode == "gt":
            future = self._demo_future(start_idx)
        elif self.future_mode in ("gen", "shift"):
            future = self._generated_future()
            if future is None:
                raise FileNotFoundError(
                    "Generated future is missing; GenFuture evaluation "
                    "forbids demonstration fallback"
                )
        else:
            raise ValueError("Unknown future mode: {}".format(self.future_mode))
        if self.future_mode == "shift" or self.future_shift != 0:
            future = shift_future(future, self.future_shift)
        return np.asarray(future, dtype=np.uint8)

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
        for _ in range(self.policy.obs_horizon):
            self.obs_static_hist.append(init_static.copy())
            self.obs_gripper_hist.append(init_gripper.copy())
        return {"task": self.task, "segment_id": int(self.current_segment["segment_id"])}

    def policy_input(self):
        return (
            np.stack(list(self.obs_static_hist), axis=0).astype(np.uint8),
            np.stack(list(self.obs_gripper_hist), axis=0).astype(np.uint8),
            self.future_static.astype(np.uint8),
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action[:ARM_ACTION_DIM] = np.clip(action[:ARM_ACTION_DIM], -1.0, 1.0)
        action[ARM_ACTION_DIM] = 1.0 if float(action[ARM_ACTION_DIM]) >= 0.0 else -1.0
        try:
            step_res = self.env.step(action)
        except AssertionError:
            return True, -20.0, {"success": False, "broken": True, "task": self.task, "segment_id": int(self.current_segment["segment_id"])}
        if len(step_res) == 5:
            obs, reward, terminated_raw, truncated_raw, _ = step_res
            env_done = bool(terminated_raw) or bool(truncated_raw)
        else:
            obs, reward, env_done, _ = step_res
        self.last_obs = obs
        self.obs_static_hist.append(resize_if_needed(get_u8(obs, "rgb_static"), IMAGE_SIZE))
        self.obs_gripper_hist.append(resize_if_needed(get_u8(obs, "rgb_gripper"), IMAGE_SIZE))
        curr_info = self.env.get_info()
        success = oracle_success(self.tasks_oracle, self.start_info, curr_info, self.task)
        return bool(success) or bool(env_done), float(reward), {"success": bool(success), "broken": False, "task": self.task, "segment_id": int(self.current_segment["segment_id"])}

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
        self.tasks_oracle = None


def run_task(eval_segments, policy, cache, task, future_mode, future_shift, seed, num_episodes, max_episode_steps, show_gui=False, save_video=False):
    set_seed(seed)
    runner = SFPFutureRunner(eval_segments, policy, cache, task, future_mode=future_mode, future_shift=future_shift, show_gui=show_gui)
    rows = []
    try:
        for ep in range(int(num_episodes)):
            info = runner.reset(ep)
            success = False
            total_return = 0.0
            steps = 0
            frames = []
            fr = runner.frame()
            if fr is not None:
                frames.append(fr)
            while not success and steps < int(max_episode_steps):
                obs_static, obs_gripper, future_static = runner.policy_input()
                actions = policy.action_chunk(obs_static, obs_gripper, future_static, task, goal_type=3)
                for action in actions:
                    done, reward, info = runner.step(action)
                    total_return += float(reward)
                    steps += 1
                    success = bool(info.get("success", False))
                    fr = runner.frame()
                    if fr is not None:
                        frames.append(fr)
                    if done or success or steps >= int(max_episode_steps):
                        break
            if save_video and len(frames) > 0:
                write_video(frames, OUTPUT_DIR / "sfp_{:s}_ep{:02d}.mp4".format(task, ep + 1), VIDEO_FPS)
            rows.append({
                "episode": ep + 1,
                "task": task,
                "success": bool(success),
                "steps": int(steps),
                "return": float(total_return),
                "segment_id": int(info.get("segment_id", -1)),
            })
    finally:
        runner.close()
        try:
            p.disconnect()
        except Exception:
            pass
    return {
        "success_rate": float(sum(int(r["success"]) for r in rows) / max(len(rows), 1)),
        "mean_return": float(np.mean([r["return"] for r in rows])) if rows else 0.0,
        "episodes": rows,
    }


def write_csv(path, rows):
    path = output_path(path)
    ensure_dir(path.parent)
    fieldnames = ["task", "future_mode", "future_shift", "seed", "max_episode_steps", "num_episodes", "success_rate", "mean_return"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFP on CALVIN tasks")
    parser.add_argument("--checkpoint_path", default=str(SFP_CKPT_PATH))
    parser.add_argument("--task", default="all")
    parser.add_argument("--future_mode", choices=["nofuture", "gt", "gen", "shift"], default="gen")
    parser.add_argument("--future_shift", type=int, choices=[-6, -4, -2, 0, 2, 4, 6], default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max_episode_steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--out_csv", default="results/sfp_eval.csv")
    parser.add_argument("--show_gui", type=int, choices=[0, 1], default=0)
    parser.add_argument("--save_video", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    if CALVIN_ROOT is None:
        raise RuntimeError("Set CALVIN_ROOT before running CALVIN SFP inference")
    if DATA_ROOT is None:
        raise RuntimeError("Set CALVIN_DATA_ROOT or CALVIN_ROOT before running CALVIN SFP inference")
    os.chdir(CALVIN_ROOT)
    os.environ["CALVIN_ROOT"] = str(CALVIN_ROOT)

    ckpt_path = Path(args.checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError("Missing SFP checkpoint: {}".format(ckpt_path))
    policy = SFPPolicy(ckpt_path, DEVICE)
    segments, tasks, _ = load_segments(SEGMENTS_JSON)
    _, eval_segments = split_segments(segments, TRAIN_SPLIT, int(args.seed))
    cache = EpisodeCache(DATA_ROOT)
    selected_tasks = list(tasks) if args.task == "all" else [args.task]

    rows = []
    for task in selected_tasks:
        if task not in policy.task_to_id:
            raise ValueError("Task {} is not in SFP checkpoint task list: {}".format(task, policy.tasks))
        metrics = run_task(
            eval_segments,
            policy,
            cache,
            task,
            args.future_mode,
            args.future_shift,
            args.seed,
            args.num_episodes,
            args.max_episode_steps,
            show_gui=bool(args.show_gui),
            save_video=bool(args.save_video),
        )
        row = {
            "task": task,
            "future_mode": normalize_future_mode(args.future_mode),
            "future_shift": int(args.future_shift),
            "seed": int(args.seed),
            "max_episode_steps": int(args.max_episode_steps),
            "num_episodes": int(args.num_episodes),
            "success_rate": metrics["success_rate"],
            "mean_return": metrics["mean_return"],
        }
        rows.append(row)
        print("task={} success_rate={:.3f} mean_return={:.3f}".format(task, row["success_rate"], row["mean_return"]))
    out_path = write_csv(args.out_csv, rows)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
