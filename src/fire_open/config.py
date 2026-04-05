from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "model": {
        "base_model": "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
        "adapter_path": None,
        "torch_dtype": "float16",
        "freeze_base_model": True,
        "lora": {
            "enabled": True,
            "r": 64,
            "alpha": 128,
            "dropout": 0.1,
            "target_modules": [
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        },
    },
    "data": {
        "image_root": None,
        "train_metadata": None,
        "query_metadata": None,
        "gallery_metadata": None,
        "task": "custom_jsonl",
        "split": "val",
        "dress_type": None,
        "max_short_edge": 380,
        "num_workers": 4,
    },
    "prompts": {
        "train_query": "Given the reference image <image> but modify it with: {modification}. Describe the new image ? <|end|> <|end|> <|end|> <|end|> <|end|>",
        "train_target": "{target_caption} . <|end|> <|end|> <|end|> <|end|> <|end|>",
        "gallery": "Describe the image <image> ? <|end|> <|end|> <|end|> <|end|> <|end|>",
        "eval_query": "Given the reference image <image> but modify it with: {modification}. Describe the new image ? <|end|> <|end|> <|end|> <|end|> <|end|>",
    },
    "training": {
        "output_dir": "outputs/fire_stage2",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 2,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 3000,
        "logging_steps": 50,
        "save_every_steps": 1000,
        "max_length": 512,
        "fp16": True,
        "loss_scale": 100.0,
        "recall_loss_weight_at_1": 0.4,
        "recall_loss_weight_at_5": 0.15,
    },
    "eval": {
        "batch_size": 16,
        "max_length": 512,
        "ks": [1, 5, 10, 50],
        "exclude_reference": True,
    },
}


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    _deep_update(cfg, user_cfg)
    return cfg
