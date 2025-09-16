"""Evaluation, analysis and plotting utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .train import SimpleClassifier, SimpleDataset

__all__ = ["evaluate"]


def _plot_logits_distribution(logits_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = torch.load(logits_path)
    plt.hist(logits.flatten().cpu().numpy(), bins=50)
    img_path = out_dir / "logits_hist.png"
    plt.savefig(img_path)
    print(f"Saved histogram to {img_path}")


def evaluate(ckpt_path: Path, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Load a checkpoint, run it on a synthetic validation set and print metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device)
    model_cfg = payload["cfg"]["model"]
    model = SimpleClassifier(model_cfg["input_dim"], model_cfg["num_classes"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    val_ds = SimpleDataset(
        num_samples=cfg["evaluation"]["num_samples"],
        input_dim=model_cfg["input_dim"],
        num_classes=model_cfg["num_classes"],
    )
    loader = DataLoader(val_ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False)

    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze(1)
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    accuracy = correct / total if total else 0.0

    results = {"accuracy": accuracy}

    # Dump to research folder and STDOUT for verification
    research_dir = ckpt_path.parent
    json_path = research_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Optional plotting for qualitative inspection
    _plot_logits_distribution(torch.randn(100).float(), research_dir / "images")
    return results
