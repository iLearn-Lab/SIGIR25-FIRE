from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from fire_open.config import load_config
from fire_open.modeling import FiREModelWrapper
from fire_open.trainer import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FiRE in an open-source friendly setup.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = FiREModelWrapper(cfg)
    output_dir = train(cfg, model)
    print(json.dumps({'status': 'ok', 'output_dir': str(output_dir)}, ensure_ascii=False))
