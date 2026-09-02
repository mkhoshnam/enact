# ENACT

LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation

ENACT studies short-horizon, task-consistent future videos as structured priors for robot manipulation, behavior cloning, and reinforcement-learning fine-tuning.

Project page: https://enact2026.github.io/

## Overview

The released CALVIN pipeline is:

```text
task command
    -> ontology-guided task grounding
    -> robot-free future-video generation and inpainting
    -> future-conditioned visual transformer policy
    -> BC initialization
    -> sparse-reward TD3 fine-tuning
```

The RL actor and twin critics are visual-only. Their inputs are frozen visual-policy features, the future latent, the BC base action, and the residual action. Robot/scene state is used only to initialize a CALVIN benchmark episode; it is not stored in replay or passed to the actor or critics.

The reward is sparse:

```text
step penalty + residual-action squared-L2 penalty
             + success bonus OR timeout penalty
```

`oracle_success(...)` reads the CALVIN task oracle to produce the sparse success label. This is benchmark supervision and never enters the controller as an observation.

## Repository layout

```text
ontology/
  calvin_task_ontology.json
scripts/
  calvin_build_bc_dataset.py
  calvin_train_bc.py
  calvin_train_fine_tuning_rl.py       # visual-only GenFuture/NoFuture RL
  calvin_infer_llm_future_bc_rl.py     # visual-only evaluation
  calvin_infer_gt_future_oracle.py     # explicitly labelled demo oracle
  SFP_training.py
  SFP_test.py
```

The CALVIN release supports `open_drawer`, `close_drawer`, `push_into_drawer`, `turn_on_led`, `turn_off_led`, `turn_on_lightbulb`, `turn_off_lightbulb`, and `move_slider_left`.

## Setup

Install CALVIN separately and use the same Python environment as that installation. Then install the additional dependencies and configure the paths:

```bash
pip install -r requirements.txt

export CALVIN_ROOT=<calvin-root>
export ENACT_CALVIN_OUT_BASE=<enact-calvin-outputs>
export CALVIN_GENERATED_FUTURE_ROOT=<generated-inpainted-calvin-futures>
```

`CALVIN_DATA_ROOT`, `CALVIN_SEGMENTS_JSON`, `CALVIN_BC_CKPT_PATH`, `CALVIN_RESULTS_ROOT`, and `CALVIN_FINE_TUNING_ACTOR_PATH` can override individual inputs. Without `ENACT_CALVIN_OUT_BASE`, artifacts are written under the repo-local `outputs/` directory.

For optional LLM task grounding, set `OPENAI_API_KEY`. Without it, inference uses deterministic ontology matching.

## Training

Build the BC dataset and train the visual transformer:

```bash
python scripts/calvin_build_bc_dataset.py
python scripts/calvin_train_bc.py
```

Fine-tune on generated futures with the visual-only actor and critics:

```bash
python scripts/calvin_train_fine_tuning_rl.py \
  --future_mode gen \
  --use_rafc 0 \
  --max_episode_steps 200
```

Other supported visual-only configurations are:

```bash
# Current observation repeated as a null future
python scripts/calvin_train_fine_tuning_rl.py --future_mode nofuture

# Generated future with a temporal shift and reliability-aware conditioning
python scripts/calvin_train_fine_tuning_rl.py \
  --future_mode shift --future_shift 4 --use_rafc 1
```

A requested generated future must exist. Training raises `FileNotFoundError` instead of silently substituting demonstration frames. `--future_mode gt` is also rejected by the training environment because GTFuture is an oracle evaluation condition, not a visual-only training mode.

Headless CALVIN runs use EGL by default. Set `CALVIN_USE_EGL=0` for non-EGL rendering or GUI debugging.

## Evaluation

Run visual-only GenFuture or NoFuture evaluation:

```bash
python scripts/calvin_infer_llm_future_bc_rl.py \
  --future_mode gen \
  --max_episode_steps 200

python scripts/calvin_infer_llm_future_bc_rl.py --future_mode nofuture
```

Missing generated video is a hard error at inference as well. There is no hidden demonstration fallback.

Reproduce the visual-only portions of the paper tables:

```bash
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode table2
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode fig4_table3
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode table4
```

### GTFuture oracle evaluation

GTFuture reads held-out demonstration frames. It is isolated behind a clearly labelled entry point and its outputs use separate oracle filenames:

```bash
python scripts/calvin_infer_gt_future_oracle.py --eval_mode single
python scripts/calvin_infer_gt_future_oracle.py --eval_mode table2
python scripts/calvin_infer_gt_future_oracle.py --eval_mode fig4_table3
```

Do not report GTFuture as a deployable visual-only condition.

## Reproducibility and compatibility

Both training and inference default to 200 environment steps per episode. The scripts expose seeds, checkpoint paths, future modes, temporal shifts, and result CSV paths through command-line arguments.

This revision changes the critic inputs, replay-buffer schema, and reward. Existing BC+RL and RAFC checkpoints were trained under the earlier privileged-critic/dense-reward setup and are not valid evidence for this formulation. Retrain all RL and RAFC runs before reporting new results. The frozen BC checkpoint remains the initialization.

Run the lightweight source-contract and syntax checks with:

```bash
python -m unittest discover -s tests -v
python -m py_compile \
  scripts/calvin_train_fine_tuning_rl.py \
  scripts/calvin_infer_llm_future_bc_rl.py \
  scripts/calvin_infer_gt_future_oracle.py
```

## Notes

This is a simulation-based research-code release. It does not claim real-world transfer or full end-to-end scene reconstruction from raw sensory input.

## Citation

```bibtex
@misc{khoshnazar2026enact,
  title  = {LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation},
  author = {Khoshnazar, Mohammad and Melnik, Andrew and Beetz, Michael},
  year   = {2026},
  url    = {https://enact2026.github.io/}
}
```
