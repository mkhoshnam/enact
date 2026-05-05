# ENACT CALVIN

This repository contains the CALVIN-side research code for future-conditioned manipulation policies.

The pipeline is:

```text
CALVIN demonstrations
→ future-conditioned BC dataset
→ BC policy training
→ RL fine-tuning
→ language command + ontology
→ generated/inpainted future video
→ future-conditioned policy execution
```

CALVIN is treated as an external benchmark. Install CALVIN separately and point the scripts to your local installation.

## Repository layout

```text
.
├── scripts/
│   ├── calvin_build_bc_dataset.py
│   ├── calvin_train_bc.py
│   ├── calvin_train_fine_tuning_rl.py
│   └── calvin_infer_llm_future_bc_rl.py
├── ontology/
│   └── calvin_task_ontology.json
├── requirements.txt
└── .gitignore
```

## Setup

Create or activate your CALVIN environment, then install the extra packages used by these scripts.

```bash
pip install -r requirements.txt
```

Edit the placeholder paths in the scripts:

```python
CALVIN_ROOT = Path("/path/to/calvin")
OUT_BASE = Path("/path/to/enact_calvin_outputs")
FUTURE_VIDEO_ROOT = Path("/path/to/generated_inpainted_calvin_futures")
```

For real LLM reasoning, set:

```bash
export OPENAI_API_KEY="your_key_here"
```

If the key is not set, the inference script uses an ontology-based fallback matcher.

## Build the dataset

```bash
python scripts/calvin_build_bc_dataset.py
```

This creates:

```text
/path/to/enact_calvin_outputs/calvin/training_dataset_future_bc.zarr
/path/to/enact_calvin_outputs/calvin/segments_future_bc.json
/path/to/enact_calvin_outputs/calvin/dataset_info_future_bc.json
```

## Train BC

```bash
python scripts/calvin_train_bc.py
```

This creates:

```text
/path/to/enact_calvin_outputs/calvin_bc/bc_actor_best.pt
```

## Fine-tune with RL

```bash
python scripts/calvin_train_fine_tuning_rl.py
```

This creates:

```text
/path/to/enact_calvin_outputs/calvin_fine_tuning_rl/fine_tuning_actor_best.pt
/path/to/enact_calvin_outputs/calvin_fine_tuning_rl/fine_tuning_actor_final.pt
```

## Run language + future-conditioned inference

Put the ontology here:

```text
/path/to/enact_calvin_outputs/ontology/calvin_task_ontology.json
```

Put generated/inpainted future videos using the paths referenced in the ontology, for example:

```text
/path/to/generated_inpainted_calvin_futures/turn_on_led/inpainted_robot_future.mp4
```

Then run:

```bash
python scripts/calvin_infer_llm_future_bc_rl.py
```

Example command:

```text
turn on the green led
```

The inference script saves:

```text
/path/to/enact_calvin_outputs/llm_future_policy_runs/llm_plan.json
/path/to/enact_calvin_outputs/llm_future_policy_runs/run_summary.json
```

## Ontology

The ontology file contains object classes, interaction parts, affordances, task aliases, success criteria, captions, and future-video paths. The LLM reads this ontology and maps a natural-language command to one CALVIN task key.

Example:

```text
turn on the green led
→ turn_on_led
→ led_button_surface
→ press the LED button
→ generated future video for turn_on_led
```

## Notes

This is research code. The scripts assume a working CALVIN installation and generated future videos prepared separately.
