import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc


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

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default=None):
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is None:
        return None
    return Path(default).expanduser()


CALVIN_ROOT = env_path("CALVIN_ROOT")
DATA_ROOT = env_path(
    "CALVIN_DATA_ROOT",
    CALVIN_ROOT / "dataset/task_D_D/training" if CALVIN_ROOT is not None else None,
)
OUT_BASE = env_path("ENACT_CALVIN_OUT_BASE", REPO_ROOT / "outputs")
OUT_DIR = OUT_BASE / "calvin"
OUT_ZARR = OUT_DIR / "training_dataset_future_bc.zarr"
OUT_SEGMENTS = OUT_DIR / "segments_future_bc.json"
OUT_INFO = OUT_DIR / "dataset_info_future_bc.json"

STATIC_CAMERA_KEY = "rgb_static"
GRIPPER_CAMERA_KEY = "rgb_gripper"
ACTION_KEY_PREFERENCE = "rel_actions"
OBS_HORIZON = 4
NUM_FUTURE_FRAMES = 8
ACTION_CHUNK_HORIZON = 8
FINAL_ONLY = True
FLUSH_BATCH = 128

GOAL_TYPE_MAP = {0: "near", 1: "mid", 2: "far", 3: "final"}
GOAL_TYPES = [3] if FINAL_ONLY else [0, 1, 2, 3]
NEAR_OFFSET = 4
MID_OFFSET = 8
FAR_OFFSET = 16

IMAGE_COMPRESSOR = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
NUM_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
_EP_RE = re.compile(r"episode_(\d+)\.npz$")


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_lang_annotations(data_root):
    ann_path = data_root / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        raise FileNotFoundError("auto_lang_ann.npy not found: {}".format(ann_path))
    return np.load(str(ann_path), allow_pickle=True).item(), ann_path


def build_episode_file_map(data_root):
    files = sorted(data_root.glob("episode_*.npz"))
    if len(files) == 0:
        raise RuntimeError("No episode_*.npz files found in {}".format(data_root))
    out = {}
    for p in files:
        m = _EP_RE.search(p.name)
        if m is not None:
            out[int(m.group(1))] = p
    if len(out) == 0:
        raise RuntimeError("Could not parse episode indices in {}".format(data_root))
    return out


class EpisodeCache(object):
    def __init__(self, episode_file_map):
        self.episode_file_map = episode_file_map
        self.cache = {}
        self.action_key = None

    def _action_key(self, item):
        if ACTION_KEY_PREFERENCE in item:
            return ACTION_KEY_PREFERENCE
        if "rel_actions" in item:
            return "rel_actions"
        if "actions" in item:
            return "actions"
        raise KeyError("No action key. Available: {}".format(list(item.keys())))

    def get(self, idx):
        idx = int(idx)
        if idx not in self.cache:
            if idx not in self.episode_file_map:
                raise KeyError("Episode index {} not found".format(idx))
            raw = np.load(str(self.episode_file_map[idx]), allow_pickle=True)
            item = {k: raw[k] for k in raw.files}
            if self.action_key is None:
                self.action_key = self._action_key(item)
            if STATIC_CAMERA_KEY not in item or GRIPPER_CAMERA_KEY not in item:
                raise KeyError("Missing camera keys in {}".format(self.episode_file_map[idx]))
            self.cache[idx] = item
        return self.cache[idx]


def normalize_task_name(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def build_segments(lang_ann):
    tasks = list(lang_ann["language"]["task"])
    anns = list(lang_ann["language"].get("ann", [""] * len(tasks)))
    idx_pairs = np.asarray(lang_ann["info"]["indx"], dtype=np.int32)

    task_to_id = {t: i for i, t in enumerate(TASKS)}
    segments = []
    for ann_id, (task_raw, ann, pair) in enumerate(zip(tasks, anns, idx_pairs)):
        task = normalize_task_name(task_raw)
        if task not in task_to_id:
            continue
        start_idx = int(pair[0])
        end_idx = int(pair[1])
        if end_idx <= start_idx:
            continue
        segments.append({
            "segment_id": len(segments),
            "task": task,
            "task_id": int(task_to_id[task]),
            "language": str(ann),
            "ann_id": int(ann_id),
            "global_start_idx": start_idx,
            "global_end_idx": end_idx,
            "num_actions": int(end_idx - start_idx),
            "demo_key": "{}__ann_{:06d}__{}_{}".format(task, ann_id, start_idx, end_idx),
        })
    return segments, task_to_id


def build_obs_indices(cur_idx, start_idx, obs_horizon):
    out = []
    for k in range(obs_horizon):
        v = cur_idx - (obs_horizon - 1) + k
        out.append(max(start_idx, v))
    return np.asarray(out, dtype=np.int32)


def choose_goal_idx(cur_idx, end_idx, goal_type):
    if goal_type == 0:
        return min(cur_idx + NEAR_OFFSET, end_idx)
    if goal_type == 1:
        return min(cur_idx + MID_OFFSET, end_idx)
    if goal_type == 2:
        return min(cur_idx + FAR_OFFSET, end_idx)
    return int(end_idx)


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


def sample_action_chunk(cache, cur_idx, end_idx, horizon):
    chunk = []
    last_valid = max(cur_idx, end_idx - 1)
    for k in range(horizon):
        idx = min(cur_idx + k, last_valid)
        item = cache.get(idx)
        chunk.append(np.asarray(item[cache.action_key], dtype=np.float32).reshape(-1))
    return np.stack(chunk, axis=0)


def resize_first_dim(arr, new_n):
    arr.resize((new_n, *arr.shape[1:]))


def flush_batch(start_row, batch, arrays):
    n = len(batch["task_id"])
    if n == 0:
        return start_row
    end_row = start_row + n
    for key, arr in arrays.items():
        if key in batch:
            arr[start_row:end_row] = np.asarray(batch[key])
    for key in batch:
        batch[key] = []
    return end_row


def main():
    ensure_dir(OUT_DIR)
    if DATA_ROOT is None:
        raise RuntimeError("Set CALVIN_DATA_ROOT or CALVIN_ROOT before building the CALVIN BC dataset")
    if not DATA_ROOT.exists():
        raise FileNotFoundError("DATA_ROOT not found: {}".format(DATA_ROOT))

    lang_ann, ann_path = load_lang_annotations(DATA_ROOT)
    episode_map = build_episode_file_map(DATA_ROOT)
    cache = EpisodeCache(episode_map)
    segments, task_to_id = build_segments(lang_ann)

    if len(segments) == 0:
        raise RuntimeError("No supported segments found in {}".format(ann_path))

    first = cache.get(segments[0]["global_start_idx"])
    static0 = np.asarray(first[STATIC_CAMERA_KEY], dtype=np.uint8)
    gripper0 = np.asarray(first[GRIPPER_CAMERA_KEY], dtype=np.uint8)
    action0 = np.asarray(first[cache.action_key], dtype=np.float32).reshape(-1)

    sh, sw, sc = static0.shape
    gh, gw, gc = gripper0.shape
    action_dim = int(action0.shape[0])
    arm_dim = max(1, action_dim - 1)

    total_samples = 0
    for seg in segments:
        total_samples += len(GOAL_TYPES) * max(0, int(seg["global_end_idx"]) - int(seg["global_start_idx"]))

    if OUT_ZARR.exists():
        shutil.rmtree(OUT_ZARR)

    print("building dataset")
    print("tasks:", TASKS)
    print("segments:", len(segments), "samples:", total_samples)
    print("output:", OUT_ZARR)

    z = zarr.open(str(OUT_ZARR), mode="w")
    z.attrs["tasks"] = TASKS
    z.attrs["task_to_id"] = task_to_id
    z.attrs["calvin_root"] = str(CALVIN_ROOT) if CALVIN_ROOT is not None else ""
    z.attrs["data_root"] = str(DATA_ROOT)
    z.attrs["obs_horizon"] = OBS_HORIZON
    z.attrs["num_future_frames"] = NUM_FUTURE_FRAMES
    z.attrs["action_chunk_horizon"] = ACTION_CHUNK_HORIZON
    z.attrs["action_dim"] = action_dim
    z.attrs["arm_dim"] = arm_dim
    z.attrs["action_key"] = cache.action_key

    arrays = {
        "obs_static_frames": z.create_dataset("obs_static_frames", shape=(0, OBS_HORIZON, sh, sw, sc), chunks=(32, OBS_HORIZON, sh, sw, sc), dtype="u1", compressor=IMAGE_COMPRESSOR, overwrite=True),
        "obs_gripper_frames": z.create_dataset("obs_gripper_frames", shape=(0, OBS_HORIZON, gh, gw, gc), chunks=(32, OBS_HORIZON, gh, gw, gc), dtype="u1", compressor=IMAGE_COMPRESSOR, overwrite=True),
        "future_static_frames": z.create_dataset("future_static_frames", shape=(0, NUM_FUTURE_FRAMES, sh, sw, sc), chunks=(32, NUM_FUTURE_FRAMES, sh, sw, sc), dtype="u1", compressor=IMAGE_COMPRESSOR, overwrite=True),
        "action_chunk": z.create_dataset("action_chunk", shape=(0, ACTION_CHUNK_HORIZON, arm_dim), chunks=(2048, ACTION_CHUNK_HORIZON, arm_dim), dtype="f4", compressor=NUM_COMPRESSOR, overwrite=True),
        "gripper_chunk": z.create_dataset("gripper_chunk", shape=(0, ACTION_CHUNK_HORIZON), chunks=(2048, ACTION_CHUNK_HORIZON), dtype="f4", compressor=NUM_COMPRESSOR, overwrite=True),
        "task_id": z.create_dataset("task_id", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "goal_type": z.create_dataset("goal_type", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "goal_step": z.create_dataset("goal_step", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "segment_id": z.create_dataset("segment_id", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "step_idx": z.create_dataset("step_idx", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "global_idx": z.create_dataset("global_idx", shape=(0,), chunks=(4096,), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "obs_indices": z.create_dataset("obs_indices", shape=(0, OBS_HORIZON), chunks=(4096, OBS_HORIZON), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
        "future_indices": z.create_dataset("future_indices", shape=(0, NUM_FUTURE_FRAMES), chunks=(4096, NUM_FUTURE_FRAMES), dtype="i4", compressor=NUM_COMPRESSOR, overwrite=True),
    }

    batch = {k: [] for k in arrays.keys()}
    arm_sum = np.zeros((arm_dim,), dtype=np.float64)
    arm_sq_sum = np.zeros((arm_dim,), dtype=np.float64)
    total_action_rows = 0
    row = 0
    seg_meta = []

    for seg in segments:
        num_written = 0
        start_idx = int(seg["global_start_idx"])
        end_idx = int(seg["global_end_idx"])
        for cur_idx in range(start_idx, end_idx):
            obs_idx = build_obs_indices(cur_idx, start_idx, OBS_HORIZON)
            obs_static = np.stack([np.asarray(cache.get(int(j))[STATIC_CAMERA_KEY], dtype=np.uint8) for j in obs_idx], axis=0)
            obs_gripper = np.stack([np.asarray(cache.get(int(j))[GRIPPER_CAMERA_KEY], dtype=np.uint8) for j in obs_idx], axis=0)

            for goal_type in GOAL_TYPES:
                goal_idx = choose_goal_idx(cur_idx, end_idx, goal_type)
                fut_idx = sample_future_indices(cur_idx, goal_idx, NUM_FUTURE_FRAMES)
                future_static = np.stack([np.asarray(cache.get(int(j))[STATIC_CAMERA_KEY], dtype=np.uint8) for j in fut_idx], axis=0)
                action_full = sample_action_chunk(cache, cur_idx, end_idx, ACTION_CHUNK_HORIZON)
                arm = action_full[:, :arm_dim].astype(np.float32)
                grip = (action_full[:, arm_dim] > 0).astype(np.float32) if action_dim > arm_dim else np.zeros((ACTION_CHUNK_HORIZON,), dtype=np.float32)

                batch["obs_static_frames"].append(obs_static)
                batch["obs_gripper_frames"].append(obs_gripper)
                batch["future_static_frames"].append(future_static)
                batch["action_chunk"].append(arm)
                batch["gripper_chunk"].append(grip)
                batch["task_id"].append(int(seg["task_id"]))
                batch["goal_type"].append(int(goal_type))
                batch["goal_step"].append(int(goal_idx))
                batch["segment_id"].append(int(seg["segment_id"]))
                batch["step_idx"].append(int(cur_idx - start_idx))
                batch["global_idx"].append(int(cur_idx))
                batch["obs_indices"].append(obs_idx)
                batch["future_indices"].append(fut_idx)

                arm_sum += arm.sum(axis=0, dtype=np.float64)
                arm_sq_sum += np.square(arm, dtype=np.float64).sum(axis=0, dtype=np.float64)
                total_action_rows += int(arm.shape[0])
                num_written += 1

                if len(batch["task_id"]) >= FLUSH_BATCH:
                    new_n = row + len(batch["task_id"])
                    for arr in arrays.values():
                        resize_first_dim(arr, new_n)
                    row = flush_batch(row, batch, arrays)

        s = dict(seg)
        s["num_samples"] = int(num_written)
        seg_meta.append(s)

    if len(batch["task_id"]) > 0:
        new_n = row + len(batch["task_id"])
        for arr in arrays.values():
            resize_first_dim(arr, new_n)
        row = flush_batch(row, batch, arrays)

    arm_mean = arm_sum / max(float(total_action_rows), 1.0)
    arm_var = np.maximum(arm_sq_sum / max(float(total_action_rows), 1.0) - np.square(arm_mean), 1e-8)
    arm_std = np.sqrt(arm_var)

    z.create_dataset("stats/arm_action_mean", data=arm_mean.astype(np.float32), overwrite=True)
    z.create_dataset("stats/arm_action_std", data=arm_std.astype(np.float32), overwrite=True)

    with open(OUT_SEGMENTS, "w", encoding="utf-8") as f:
        json.dump({"tasks": TASKS, "task_to_id": task_to_id, "segments": seg_meta}, f, indent=2)

    info = {
        "tasks": TASKS,
        "task_to_id": task_to_id,
        "num_segments": len(seg_meta),
        "num_samples": int(row),
        "zarr": str(OUT_ZARR),
        "segments_json": str(OUT_SEGMENTS),
        "obs_horizon": OBS_HORIZON,
        "num_future_frames": NUM_FUTURE_FRAMES,
        "action_chunk_horizon": ACTION_CHUNK_HORIZON,
        "arm_dim": arm_dim,
        "action_dim": action_dim,
    }
    with open(OUT_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("done", row)


if __name__ == "__main__":
    main()
