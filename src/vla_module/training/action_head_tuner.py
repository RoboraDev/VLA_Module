from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from vla_module.data.dataloaders import (
    LeRobotDataLoaderConfig,
    LeRobotSequenceDataset,
    create_lerobot_dataloader,
)
from vla_module.models.pi0.PI0Policy import PI0Policy


LOGGER = logging.getLogger(__name__)


@dataclass
class ActionHeadTrainingConfig:
    """Top-level configuration for frozen-VLM fine-tuning."""

    data: LeRobotDataLoaderConfig = field(default_factory=LeRobotDataLoaderConfig)
    num_epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # options: "bf16", "fp16", "none"
    log_interval: int = 50
    checkpoint_interval: int | None = None
    output_dir: str | Path | None = None
    resume_from: str | None = None
    device: str | None = None


def freeze_vlm(policy: PI0Policy) -> None:
    """Freeze the vision-language backbone while keeping the action expert trainable."""

    vlm = policy.model.paligemma_with_expert.paligemma
    for param in vlm.parameters():
        param.requires_grad = False
    vlm.eval()

    LOGGER.info("Frozen %s parameters", sum(p.numel() for p in vlm.parameters()))

    # Ensure action expert and projection heads remain trainable
    for module in (
        policy.model.paligemma_with_expert.gemma_expert,
        policy.model.action_in_proj,
        policy.model.action_out_proj,
        policy.model.state_proj,
        policy.model.action_time_mlp_in,
        policy.model.action_time_mlp_out,
    ):
        module.train()
        for param in module.parameters():
            param.requires_grad = True


class VLMFreezeTrainer:
    """Orchestrates action-head fine-tuning with a frozen VLM tower."""

    def __init__(self, policy: PI0Policy, cfg: ActionHeadTrainingConfig) -> None:
        self.policy = policy
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
            or getattr(policy.config, "device", None)
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy.to(self.device)
        freeze_vlm(self.policy)

        self.dataset: LeRobotSequenceDataset
        self.dataloader = self._build_dataloader()
        self.opt_cfg = self.policy.config.get_optimizer_preset()
        self.optimizer = self._build_optimizer()
        self.grad_clip = self.opt_cfg.grad_clip_norm
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self._use_fp16)

        self.total_steps = self._compute_total_steps()
        self.start_step = 0

        if cfg.resume_from:
            self._restore_checkpoint(Path(cfg.resume_from))

    @property
    def _use_fp16(self) -> bool:
        return self.cfg.mixed_precision.lower() == "fp16" and self.device.type == "cuda"

    @property
    def _autocast_enabled(self) -> bool:
        return self.device.type == "cuda" and self.cfg.mixed_precision.lower() in {"bf16", "fp16"}

    @property
    def _autocast_dtype(self) -> torch.dtype:
        if self.cfg.mixed_precision.lower() == "bf16":
            return torch.bfloat16
        if self.cfg.mixed_precision.lower() == "fp16":
            return torch.float16
        return torch.float32

    def _build_dataloader(self):
        self.dataset, dataloader = create_lerobot_dataloader(self.policy.config, self.cfg.data)
        LOGGER.info(
            "Loaded dataset %s with %d frames (sequences: %d)",
            self.cfg.data.repo_id,
            self.dataset.base.num_frames,
            len(self.dataset),
        )
        return dataloader

    def _build_optimizer(self):
        params = [p for p in self.policy.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters detected after freezing VLM.")
        optimizer = self.opt_cfg.build(params)
        LOGGER.info(
            "Initialized optimizer %s with lr=%g (trainable params=%d)",
            optimizer.__class__.__name__,
            self.opt_cfg.lr,
            sum(p.numel() for p in params),
        )
        return optimizer

    def _build_scheduler(self):
        sched_cfg = self.policy.config.get_scheduler_preset()
        scheduler = sched_cfg.build(self.optimizer, self._compute_total_steps())
        if scheduler is not None:
            LOGGER.info("Using scheduler %s", scheduler.__class__.__name__)
        return scheduler

    def _compute_total_steps(self) -> int:
        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch == 0:
            return 1
        eff_batches = max(1, batches_per_epoch // self.cfg.gradient_accumulation_steps)
        if batches_per_epoch % self.cfg.gradient_accumulation_steps:
            eff_batches += 1
        steps = eff_batches * self.cfg.num_epochs
        if self.cfg.max_steps:
            steps = min(steps, self.cfg.max_steps)
        return max(steps, 1)

    def _restore_checkpoint(self, ckpt_path: Path) -> None:
        LOGGER.info("Loading checkpoint from %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self._use_fp16 and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_step = checkpoint.get("step", 0)
        LOGGER.info("Resumed training from global step %d", self.start_step)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, val in batch.items():
            moved[key] = val.to(self.device) if torch.is_tensor(val) else val
        return moved

    def _should_log(self, step: int) -> bool:
        return (step + 1) % self.cfg.log_interval == 0 or step == 0

    def _should_checkpoint(self, step: int) -> bool:
        return (
            self.cfg.checkpoint_interval is not None
            and (step + 1) % self.cfg.checkpoint_interval == 0
        )

    def _save_checkpoint(self, step: int) -> None:
        if not self.cfg.output_dir:
            return
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / f"checkpoint_{step + 1}.pt"
        payload = {
            "model_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step + 1,
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        if self._use_fp16:
            payload["scaler_state"] = self.scaler.state_dict()
        torch.save(payload, ckpt_path)
        LOGGER.info("Saved checkpoint to %s", ckpt_path)

    def train(self) -> None:
        LOGGER.info(
            "Starting training for %d epochs (%d total steps)",
            self.cfg.num_epochs,
            self.total_steps,
        )
        self.policy.train()

        global_step = self.start_step
        epochs_completed = 0

        for epoch in range(self.cfg.num_epochs):
            if self.cfg.max_steps and global_step >= self.cfg.max_steps:
                break

            for batch_idx, batch in enumerate(self.dataloader):
                if self.cfg.max_steps and global_step >= self.cfg.max_steps:
                    break

                batch = self._move_batch_to_device(batch)
                with autocast(enabled=self._autocast_enabled, dtype=self._autocast_dtype):
                    loss, metrics = self.policy(batch)
                    loss = loss / self.cfg.gradient_accumulation_steps

                if self._use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                    if self.grad_clip and self.grad_clip > 0:
                        if self._use_fp16:
                            self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(
                            [p for p in self.policy.parameters() if p.requires_grad],
                            self.grad_clip,
                        )

                    if self._use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if self._should_log(global_step - 1):
                        LOGGER.info(
                            "Epoch %d | Step %d | Loss %.4f | MAE %.4f | L2 %.4f",
                            epoch,
                            global_step,
                            metrics["loss"],
                            metrics.get("mae", 0.0),
                            metrics.get("l2", 0.0),
                        )

                    if self._should_checkpoint(global_step - 1):
                        self._save_checkpoint(global_step - 1)

            epochs_completed += 1

        LOGGER.info("Training complete. Completed epochs: %d, total steps: %d", epochs_completed, global_step)


def run_action_head_training(policy: PI0Policy, cfg: ActionHeadTrainingConfig) -> VLMFreezeTrainer:
    """Convenience function to instantiate and run the trainer."""

    trainer = VLMFreezeTrainer(policy, cfg)
    trainer.train()
    return trainer


__all__ = [
    "ActionHeadTrainingConfig",
    "VLMFreezeTrainer",
    "freeze_vlm",
    "run_action_head_training",
]
