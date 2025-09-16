"""Entry-point orchestrating smoke-test and full experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

from .evaluate import evaluate
from .train import train

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cfg_path: Path) -> None:
    cfg = _load_yaml(cfg_path)
    print(f"Loaded configuration from {cfg_path}")
    ckpt_path = train(cfg)
    evaluate(ckpt_path, cfg)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="AIS-KD Experimental Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run quick validation")
    group.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    args = parser.parse_args()

    if args.smoke_test:
        cfg_file = _CONFIG_DIR / "smoke_test.yaml"
    else:
        cfg_file = _CONFIG_DIR / "full_experiment.yaml"

    try:
        _run(cfg_file)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
