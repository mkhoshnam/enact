"""Every experimental constant the paper states, in one place.

Any script that produces a number appearing in the paper imports from here.
The point is that a horizon or seed count can never silently differ between
Table I and Table VIII, and that changing a reported protocol value requires
editing exactly one line.

Values below are transcribed from the manuscript. `describe()` prints them so a
run log records what protocol it used.
"""

from __future__ import annotations

# --- Benchmarks -------------------------------------------------------------

# Sec. IV-A: eight CALVIN tasks. Names match the repo's segment keys.
CALVIN_TASKS = (
    "open_drawer",
    "close_drawer",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "push_into_drawer",
    "move_slider_left",
)

# Sec. IV-A / Table IV: five RoboCasa tasks.
ROBOCASA_TASKS = (
    "CloseDrawer",
    "OpenSingleDoor",
    "OpenDrawer",
    "TurnOnMicrowave",
    "PickPlaceSinkToCounter",
)

# --- Evaluation protocol ----------------------------------------------------

# Sec. IV-B: "three independent seeds (42, 43, 44)".
SEEDS = (42, 43, 44)

# Sec. IV-B: "300 simulation episodes per task and seed; episodes last at most
# 200 steps". NOTE: the repo README still says 150 and is stale.
NUM_EVAL_EPISODES = 300
MAX_EPISODE_STEPS = 200

# Sec. IV-B: "140k environment steps".
TOTAL_ENV_STEPS = 140_000

# Sec. V-E / Table V: 20 trials per task per method on hardware.
REAL_ROBOT_TRIALS_PER_TASK = 20
REAL_ROBOT_TASKS = ("open_kettle", "close_kettle", "close_microwave_door")
REAL_ROBOT_METHODS = ("nofuture", "fec", "rafc")

# --- Future interface -------------------------------------------------------

# Sec. IV-B: "Each future contains T = 16 frames and is compressed into B = 4
# temporal bins."
FUTURE_FRAMES = 16
FUTURE_BINS = 4

# Sec. III-B, Eq. (5): 28-frame source rollout, 16-frame window, base offset 6.
SOURCE_ROLLOUT_FRAMES = 28
WINDOW_BASE_OFFSET = 6

# --- Perturbations ----------------------------------------------------------

# Sec. III-C: RAFC's local candidates, constructed at train and eval time.
LOCAL_SHIFTS = (-2, 0, 2)

# Sec. III-B: global perturbations, evaluation-only.
GLOBAL_SHIFTS = (-6, -4, -2, 0, 2, 4, 6)
NONZERO_GLOBAL_SHIFTS = tuple(s for s in GLOBAL_SHIFTS if s != 0)

# Sec. III-B: monotone rate warps, evaluation-only.
RATE_WARPS = (0.75, 1.25)

# Which rounding rule maps warp step i to a source frame. The paper writes
# "round(...)" without fixing the tie rule, and the two candidates disagree at
# exact halves (rho = 0.75 hits them). This MUST be set to whatever produced
# the reported Table VIII numbers, not to whichever looks tidier.
#   "rint"     numpy.rint / Python round  -> banker's rounding, round(0.5) == 0
#   "half_up"  floor(x + 0.5)             -> round(0.5) == 1
RATE_WARP_ROUNDING = "rint"

# --- RL hyperparameters (Sec. IV-B) -----------------------------------------

BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
ACTOR_LR = 5e-5
CRITIC_LR = 5e-5
TARGET_POLICY_NOISE = 0.05
NOISE_CLIP = 0.08

# --- Reward (Sec. III-D, Eqs. 14-15) ----------------------------------------

STEP_PENALTY = -0.005
DELTA_L2 = 0.02
SUCCESS_REWARD = 120.0
TIMEOUT_PENALTY = -25.0
BROKEN_PENALTY = -20.0
REWARD_CLIP = (-50.0, 150.0)

# --- Consistency checks -----------------------------------------------------


def _check() -> None:
    assert len(set(CALVIN_TASKS)) == len(CALVIN_TASKS) == 8
    assert len(set(ROBOCASA_TASKS)) == len(ROBOCASA_TASKS) == 5
    assert len(SEEDS) == 3
    assert 0 in LOCAL_SHIFTS and 0 in GLOBAL_SHIFTS
    assert set(LOCAL_SHIFTS).issubset(set(GLOBAL_SHIFTS))
    assert len(NONZERO_GLOBAL_SHIFTS) == 6, "Avg. shifted averages six columns"
    assert RATE_WARP_ROUNDING in ("rint", "half_up")
    # Eq. (5) must fit inside the source rollout at the extreme offsets.
    for s in GLOBAL_SHIFTS:
        start = WINDOW_BASE_OFFSET + s
        assert 0 <= start and start + FUTURE_FRAMES <= SOURCE_ROLLOUT_FRAMES, (
            "shift {} does not fit a {}-frame window in a {}-frame source "
            "rollout".format(s, FUTURE_FRAMES, SOURCE_ROLLOUT_FRAMES)
        )


_check()


def describe() -> str:
    """One-line protocol fingerprint to write into every result file."""
    return (
        "seeds={seeds} episodes={eps} horizon={hor} env_steps={steps} "
        "T={T} B={B} local={loc} global={glob} rates={rates} "
        "rounding={rounding}".format(
            seeds=list(SEEDS),
            eps=NUM_EVAL_EPISODES,
            hor=MAX_EPISODE_STEPS,
            steps=TOTAL_ENV_STEPS,
            T=FUTURE_FRAMES,
            B=FUTURE_BINS,
            loc=list(LOCAL_SHIFTS),
            glob=list(GLOBAL_SHIFTS),
            rates=list(RATE_WARPS),
            rounding=RATE_WARP_ROUNDING,
        )
    )


if __name__ == "__main__":
    print(describe())
