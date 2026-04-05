# FiRE: Open-Source-Friendly Cleanup

An open-source-friendly cleanup of the FiRE codebase derived from the private scripts you provided.

This repository is intended to make the project easier to release, reproduce, and maintain. It focuses on three practical goals:
- removing hard-coded local paths,
- moving hyperparameters into explicit configs,
- keeping the default evaluation pipeline fair and easy to understand.

> This is **not** a byte-for-byte mirror of the original private project. It is a cleaned, public-facing reorganization designed for reproducibility.

---

## Highlights

- **No hard-coded absolute paths**
- **YAML-based configuration for major hyperparameters**
- **Separated training and evaluation entry points**
- **Default evaluation keeps only a fair public protocol**
- **Repository structure suitable for GitHub release**

---

## Repository Layout

```text
fire_opensource_clean/
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── configs/
│   ├── train_stage2.example.yaml
│   └── eval.example.yaml
├── docs/
│   └── cleanup_notes.md
├── scripts/
│   ├── train.py
│   └── eval.py
└── src/fire_open/
    ├── __init__.py
    ├── config.py
    ├── datasets.py
    ├── losses.py
    ├── modeling.py
    └── trainer.py
```

---

## What Was Cleaned Up

### 1. Local paths were removed
The original scripts relied on machine-specific paths such as `/home/...` and private storage layouts. In this version, all dataset and output locations are controlled through YAML config files.

Example:

```yaml
data:
  image_root: ./data/images
  train_metadata: ./data/annotations/fire_train.jsonl
```

### 2. Hyperparameters were made explicit
Key training and evaluation settings are now exposed in `configs/*.yaml`, including:
- LoRA settings (`r`, `alpha`, `dropout`)
- learning rate
- batch size
- number of epochs
- warmup steps
- loss-related weights
- decoding / max-length settings used by the wrapper

### 3. Training and evaluation were separated
Instead of mixing multiple internal workflows in a single script, this version provides two explicit entry points:
- `scripts/train.py`
- `scripts/eval.py`

This makes it easier for external users to see how the project is supposed to be run.

### 4. Default evaluation was restricted to a fair public protocol
The original codebase appears to contain branches and dependencies that are not ideal as the default public benchmark path, such as:
- private intermediate JSON files,
- private auxiliary captions or annotations,
- local precomputed vision-token caches,
- internal modes whose meaning is unclear from the outside.

In this cleaned release, the default evaluation assumes only:
- **query** = reference image + text modification
- **gallery** = candidate image pool
- **metric** = Recall@K

That choice is intentional. It reduces ambiguity and avoids accidentally introducing extra test-time information.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Train

Edit the training config first:

```bash
configs/train_stage2.example.yaml
```

Then run:

```bash
python scripts/train.py --config configs/train_stage2.example.yaml
```

### Evaluate

Edit the evaluation config first:

```bash
configs/eval.example.yaml
```

Then run:

```bash
python scripts/eval.py --config configs/eval.example.yaml
```

Example output:

```json
{
  "Recall@1": 0.23,
  "Recall@5": 0.51,
  "Recall@10": 0.64,
  "Recall@50": 0.88
}
```

---

## Configuration Overview

### Training config

The example training config exposes the main public knobs:

```yaml
seed: 42
model:
  base_model: Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5
  adapter_path: null
  freeze_base_model: true
  lora:
    enabled: true
    r: 64
    alpha: 128
    dropout: 0.1

data:
  task: custom_jsonl
  image_root: ./data/images
  train_metadata: ./data/annotations/fire_train.jsonl
  max_short_edge: 380
  num_workers: 4

training:
  output_dir: ./outputs/fire_stage2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  num_train_epochs: 2
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 3000
  logging_steps: 50
  save_every_steps: 1000
  max_length: 512
  loss_scale: 100.0
  recall_loss_weight_at_1: 0.4
  recall_loss_weight_at_5: 0.15
```

### Evaluation config

```yaml
seed: 42
model:
  base_model: Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5
  adapter_path: ./outputs/fire_stage2

data:
  task: custom_jsonl   # custom_jsonl | fashioniq | cirr
  image_root: ./data/images
  query_metadata: ./data/annotations/eval_queries.jsonl
  gallery_metadata: ./data/annotations/eval_gallery.jsonl
  split: val
  dress_type: dress
  max_short_edge: 380
  num_workers: 4

eval:
  batch_size: 16
  max_length: 512
  ks: [1, 5, 10, 50]
  exclude_reference: true
```

---

## Data Format

### Training data (`jsonl`)

Recommended format: one sample per line.

```json
{
  "sample_id": "000001",
  "reference_image": "train/ref/0001.jpg",
  "target_image": "train/tgt/0001.jpg",
  "reference_id": "ref_0001",
  "target_id": "tgt_0001",
  "modification": "change the red shirt into a blue striped shirt",
  "reference_caption": "a person wearing a plain red shirt",
  "target_caption": "a person wearing a blue striped shirt"
}
```

Required fields:
- `reference_image`
- `target_image`
- `modification`

Optional fields:
- `reference_caption`
- `target_caption`
- `reference_id`
- `target_id`
- `sample_id`

If captions are unavailable, they can be left empty, though prompt quality may be weaker.

### Evaluation query metadata (`jsonl`)

```json
{
  "query_id": "q1",
  "reference_image": "eval/ref/001.jpg",
  "reference_id": "img_001",
  "modification": "make the bag black and remove the logo",
  "target_id": "img_128",
  "exclude_ids": ["img_001"]
}
```

### Evaluation gallery metadata (`jsonl`)

```json
{
  "image_id": "img_128",
  "image_path": "eval/gallery/128.jpg"
}
```

By design, the default evaluation path does **not** consume extra target-side text such as `target_caption`.

---

## Supported Evaluation Tasks

This cleaned version includes readers for:
- `custom_jsonl`
- `fashioniq`
- `cirr`

### FashionIQ example

```yaml
data:
  task: fashioniq
  image_root: ./data/fashion_iq_data
  split: val
  dress_type: dress
```

### CIRR example

```yaml
data:
  task: cirr
  image_root: ./data/CIRR
  split: val
```

---

## Fairness Policy for Public Evaluation

The following are intentionally **not** enabled as the default public evaluation path:

1. private checkpoint locations,
2. private `hbh_*` annotation files,
3. local `.pt` caches for precomputed vision tokens,
4. internal modes such as `case`, `pre_vision`, or `classic` as the default benchmark entry,
5. additional test-time text not available in the standard retrieval setup.

This repository favors a conservative public protocol so that external users can reproduce results without reverse-engineering internal assumptions.

---

## Known Limitations

- This release prioritizes **public reproducibility** over strict fidelity to a private internal codebase.
- Some logic from the original project appears tightly coupled to private intermediate files; here it has been replaced with explicit metadata files and explicit configuration.
- Full end-to-end reproduction still depends on the user providing the correct datasets, model access, and runtime environment.

---

## Recommended Files Before Public Release

Before pushing the repository publicly, it would still be good to add:
- `LICENSE`
- dataset preparation scripts
- benchmark download instructions
- a results table for the public setting
- checkpoint release notes
- a citation block for the paper

---

## Notes

Additional cleanup rationale is documented in:

```text
docs/cleanup_notes.md
```

A Chinese version of this README is also included:

```text
README.zh-CN.md
```
