"""Training utilities.
This file contains classes and functions required for training a generic
PyTorch model.  In a normal refactor we would extract the real logic from the
monolithic script, but the provided “Experiment Code” section is empty.  In
order to keep the project runnable we therefore supply minimal, production‐
ready scaffolding that can be expanded once real code becomes available.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "ExperimentState",
    "SimpleClassifier",
    "train_one_epoch",
    "train",
]


@dataclass
class ExperimentState:
    step: int
    epoch: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class SimpleDataset(Dataset):
    """A dummy dataset that returns random tensors. Used as a placeholder."""

    def __init__(self, num_samples: int, input_dim: int, num_classes: int):
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:  # noqa: D401
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).long()
        return x, y


class SimpleClassifier(nn.Module):
    """A very small MLP for demonstration / smoke-test purposes."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimiser: optim.Optimizer, device: torch.device) -> float:  # noqa: D401,E501
    """Train for a single epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).squeeze(1)
        optimiser.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def train(cfg: Dict[str, Any]) -> Path:
    """End-to-end training loop driven by a configuration dictionary.

    The function returns the path to the checkpoint written in the research
    artefact directory so that downstream evaluation can locate it reliably.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility guard.
    torch.manual_seed(cfg.get("seed", 42))

    train_ds = SimpleDataset(
        num_samples=cfg["data"]["num_samples"],
        input_dim=cfg["model"]["input_dim"],
        num_classes=cfg["model"]["num_classes"],
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    model = SimpleClassifier(
        input_dim=cfg["model"]["input_dim"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    research_dir = Path(".research") / cfg["run_name"]
    research_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = research_dir / "checkpoint.pt"

    state = ExperimentState(step=0, epoch=0)

    for epoch in range(cfg["training"]["epochs"]):
        avg_loss = train_one_epoch(model, loader, criterion, optimiser, device)
        state.epoch = epoch + 1
        state.step += len(loader)
        print(f"Epoch {epoch+1}: avg-loss={avg_loss:.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg, "state": asdict(state)}, ckpt_path)
    print(f"Model saved to {ckpt_path}")
    # Also persist metadata for inspection
    with open(research_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return ckpt_path
