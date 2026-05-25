# ENACT

LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation

ENACT studies short-horizon, task-consistent future videos as structured priors for robot manipulation, behavior cloning, and reinforcement-learning fine-tuning.

Project page: https://enact2026.github.io/

## Overview

ENACT uses an ontology-guided LLM reasoner to ground a user command, generate or select a short-horizon future video, and condition a robot policy on that future.

Pipeline:

    task command
    -> ontology-guided LLM task grounding
    -> robot-free digital-twin future rollout
    -> robot inpainting / generated future video
    -> future-conditioned policy
    -> BC and RL fine-tuning

## Current release

This repository currently releases the CALVIN part of ENACT:

- CALVIN dataset construction
- future-conditioned BC training
- BC-initialized RL fine-tuning
- ontology/LLM-guided inference
- generated future video conditioning
- reliability-aware future conditioning for generated, wrong, and shifted futures

The RoboCasa scripts, generated futures, and trained models are part of the same ENACT project and will be released soon.

## Structure

    ontology/
      calvin_task_ontology.json

    scripts/
      calvin_build_bc_dataset.py
      calvin_train_bc.py
      calvin_train_fine_tuning_rl.py
      calvin_infer_llm_future_bc_rl.py
      SFP_training.py
      SFP_test.py

## Supported CALVIN tasks

- open_drawer
- close_drawer
- push_into_drawer
- turn_on_led
- turn_off_led
- turn_on_lightbulb
- turn_off_lightbulb
- move_slider_left

## Setup

Install CALVIN separately, then configure paths with environment variables:

    export CALVIN_ROOT=<calvin-root>
    export ENACT_CALVIN_OUT_BASE=<enact-calvin-outputs>
    export CALVIN_GENERATED_FUTURE_ROOT=<generated-inpainted-calvin-futures>

`CALVIN_DATA_ROOT`, `CALVIN_SEGMENTS_JSON`, `CALVIN_BC_CKPT_PATH`,
`CALVIN_RESULTS_ROOT`, and `CALVIN_FINE_TUNING_ACTOR_PATH` can override
individual inputs when needed. If `ENACT_CALVIN_OUT_BASE` is not set, scripts
write to the repo-local `outputs/` directory.

Install dependencies:

    pip install -r requirements.txt

For LLM task grounding:

    export OPENAI_API_KEY=your_key_here

Without an API key, inference falls back to ontology-based matching.

## Usage

Build dataset:

    python scripts/calvin_build_bc_dataset.py

Train BC:

    python scripts/calvin_train_bc.py

Fine-tune with RL:

    python scripts/calvin_train_fine_tuning_rl.py

By default the RL fine-tuning run is multitask and saves checkpoints under:

    outputs/rafc_rl_runs/multitask_step6_rafc_td3bc_seed42/policy_best.pt
    outputs/rafc_rl_runs/multitask_step6_rafc_td3bc_seed42/policy_final.pt

Use generated futures and/or disable RAFC for baseline checkpoints:

    CALVIN_USE_GENERATED_FUTURES_DURING_RL=1 python scripts/calvin_train_fine_tuning_rl.py
    CALVIN_USE_RAFC=0 python scripts/calvin_train_fine_tuning_rl.py

Headless CALVIN runs use EGL by default. Set `CALVIN_USE_EGL=0` when you need
non-EGL rendering or GUI debugging.

Run inference:

    python scripts/calvin_infer_llm_future_bc_rl.py

Example command:

    turn on the green led

For RAFC ablations, inference accepts simple environment switches:

    CALVIN_FUTURE_MODE=null python scripts/calvin_infer_llm_future_bc_rl.py
    CALVIN_FUTURE_MODE=shift CALVIN_FUTURE_SHIFT=4 python scripts/calvin_infer_llm_future_bc_rl.py
    CALVIN_FUTURE_MODE=wrong CALVIN_WRONG_FUTURE_VIDEO_PATH=<wrong-future-video.mp4> python scripts/calvin_infer_llm_future_bc_rl.py

Train and evaluate the SFP baseline:

    python scripts/SFP_training.py
    python scripts/SFP_test.py --task all --future_mode gen

### Reproducing paper results

```bash
# Table 2
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode table2

# Figure 4 and Table 3
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode fig4_table3

# Table 4
python scripts/calvin_infer_llm_future_bc_rl.py --eval_mode table4
```

All CALVIN evaluations use `max_episode_steps=150`.

## Notes

This is a simulation-based research-code release. It does not claim real-world transfer or full end-to-end scene reconstruction from raw sensory input.

## Citation

    @misc{khoshnazar2026enact,
      title  = {LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation},
      author = {Khoshnazar, Mohammad and Melnik, Andrew and Beetz, Michael},
      year   = {2026},
      url    = {https://enact2026.github.io/}
    }
