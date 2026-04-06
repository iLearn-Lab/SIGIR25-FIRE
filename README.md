# FiRE: Enhancing MLLMs with Fine-Grained Context Learning for Complex Image Retrieval

> Official implementation of **FiRE**, a fine-grained context learning framework for complex image retrieval.

**Oral @ SIGIR'25**

## Authors

**Bohan Hou**<sup>1</sup>, **Haoqiang Lin**<sup>1</sup>, **Xuemeng Song**<sup>1</sup>, **Haokun Wen**<sup>2,4</sup>, **Meng Liu**<sup>3</sup>, **Yupeng Hu**<sup>1</sup>, **Xiangyu Zhao**<sup>4</sup>\*

<sup>1</sup> Shandong University  
<sup>2</sup> Harbin Institute of Technology (Shenzhen)  
<sup>3</sup> Shandong Jianzhu University  
<sup>4</sup> City University of Hong Kong  
\* Corresponding author

## Links

- **Paper**: [`ACM Digital Library`](https://doi.org/10.1145/3726302.3729979)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/fire_opensource_clean)

> 如果后续有 project page、Hugging Face、demo 等链接，可以继续补在这里。

---

## Updates

- [07/2025] Paper accepted at SIGIR 2025
- [04/2026] Initial open-source release

---



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

## Citation

```bibtex
@inproceedings{hou2025fire,
  author    = {Bohan Hou and Haoqiang Lin and Xuemeng Song and Haokun Wen and Meng Liu and Yupeng Hu and Xiangyu Zhao},
  title     = {FiRE: Enhancing MLLMs with Fine-Grained Context Learning for Complex Image Retrieval},
  booktitle = {Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {803--812},
  publisher = {ACM},
  year      = {2025},
  doi       = {10.1145/3726302.3729979}
}
```
---

## License

This project is released under the Apache License 2.0.
