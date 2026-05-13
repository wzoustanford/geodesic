"""
Supervised (imitation-learning) training orchestrator for VLA agents.

Drives OpenVLAAgent through next-token CE on action-token positions, iterating
batches yielded by a DataLoader. Logs metrics, periodically saves checkpoints,
optionally runs validation.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader


class VLAOrchestrator:
    """Drives supervised IL training of an OpenVLAAgent.

    train_loader and (optional) val_loader yield batch dicts shaped for
    OpenVLAAgent.update / .validate:
      pixel_values, input_ids, attention_mask, labels  (+ optional dataset_names)
    """

    def __init__(
        self,
        agent: Any,
        train_loader: DataLoader,
        max_steps: int = 100_000,
        log_every: int = 10,
        save_every: int = 1000,
        val_loader: Optional[DataLoader] = None,
        val_every: int = 1000,
        val_steps: int = 20,
    ) -> None:
        self.agent = agent
        self.train_loader = train_loader
        self.max_steps = max_steps
        self.log_every = log_every
        self.save_every = save_every
        self.val_loader = val_loader
        self.val_every = val_every
        self.val_steps = val_steps

    def start(self) -> Dict[str, float]:
        """Run max_steps of supervised training. Returns final-step train metrics."""
        start = time.time()
        print("-" * 10 + " starting VLA supervised training " + "-" * 10, flush=True)
        train_iter = iter(self.train_loader)
        latest: Dict[str, float] = {}

        for step in range(1, self.max_steps + 1):
            batch, train_iter = self._next(self.train_loader, train_iter)
            latest = self.agent.update(batch)

            if step % self.log_every == 0:
                self._log_metrics(step, latest, time.time() - start, tag="train")
            if step % self.save_every == 0:
                self.agent.save(self.agent._default_path(f"step_{step}"))
            if self.val_loader is not None and step % self.val_every == 0:
                self._log_metrics(step, self.validate(), time.time() - start, tag="val")

        self.agent.save(self.agent._default_path("final"))
        print(f"\nVLA training complete in {(time.time()-start)/60:.1f}min", flush=True)
        return latest

    def validate(self) -> Dict[str, float]:
        """Mean metrics over val_steps batches from val_loader."""
        assert self.val_loader is not None, "validate() called without a val_loader"
        val_iter = iter(self.val_loader)
        accum: Dict[str, float] = {}
        for _ in range(self.val_steps):
            batch, val_iter = self._next(self.val_loader, val_iter)
            for k, v in self.agent.validate(batch).items():
                accum[k] = accum.get(k, 0.0) + v
        return {k: v / self.val_steps for k, v in accum.items()}

    @staticmethod
    def _next(loader: DataLoader, it):
        """Pull next batch, cycling when the loader exhausts."""
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    @staticmethod
    def _log_metrics(step: int, metrics: Dict[str, float], elapsed: float, tag: str) -> None:
        body = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"step {step} [{tag}] {body} | elapsed={elapsed/60:.1f}min", flush=True)
