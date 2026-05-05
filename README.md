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

The RoboCasa scripts, generated futures, and trained models are part of the same ENACT project and will be released soon.

## Structure

    ontology/
      calvin_task_ontology.json

    scripts/
      calvin_build_bc_dataset.py
      calvin_train_bc.py
      calvin_train_fine_tuning_rl.py
      calvin_infer_llm_future_bc_rl.py

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

Install CALVIN separately, then edit the placeholder paths in the scripts:

    CALVIN_ROOT = Path("/path/to/calvin")
    OUT_BASE = Path("/path/to/enact_calvin_outputs")
    FUTURE_VIDEO_ROOT = Path("/path/to/generated_inpainted_calvin_futures")

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

Run inference:

    python scripts/calvin_infer_llm_future_bc_rl.py

Example command:

    turn on the green led

## Notes

This is a simulation-based research-code release. It does not claim real-world transfer or full end-to-end scene reconstruction from raw sensory input.

## Citation

    @misc{khoshnazar2026enact,
      title  = {LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation},
      author = {Khoshnazar, Mohammad and Melnik, Andrew and Beetz, Michael},
      year   = {2026},
      url    = {https://enact2026.github.io/}
    }
