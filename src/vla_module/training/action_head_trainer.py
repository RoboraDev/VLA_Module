"""Action Head Trainer

Trainer class for action-head-only fine-tuning with frozen VLM backbone.
Supports both SmolVLA and PI0 architectures.
"""

# TODO: Shiven checkout the different functionalities implemented in this code.

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from vla_module.data.lerobot_adapter import (
    DatasetAdapterConfig,
    LeRobotDatasetAdapter,
    create_lerobot_dataloader,
)
from vla_module.training.action_space_config import (
    ActionSpaceAdapter,
    ObservationSpaceAdapter,
    create_action_space_config_from_dataset,
    create_observation_space_config_from_dataset,
)
from vla_module.training.checkpoint_utils import (
    cleanup_old_checkpoints,
    get_latest_checkpoint,
    load_checkpoint,
    load_pretrained_model,
    save_checkpoint,
)
from vla_module.training.freeze_utils import (
    print_trainable_parameters,
    setup_action_head_only_training,
)
from vla_module.training.training_config import ActionHeadTrainingConfig


class ActionHeadTrainer:
    """Trainer for action-head-only fine-tuning.
    
    This trainer:
    - Freezes the VLM backbone (vision + language model)
    - Trains only the action prediction head and optional projection layers
    - Supports SmolVLA and PI0 architectures
    - Handles gradient accumulation and mixed precision
    """
    
    def __init__(
        self,
        config: ActionHeadTrainingConfig,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the action head trainer.
        
        Args:
            config: Training configuration
            model: VLA model to train
            device: Device to train on
        """
        self.config = config
        self.device = device
        self.model = model.to(device)
        
        # Setup model for action-head-only training
        self._setup_model()
        
        # Create data loaders
        self._setup_data()
        
        # Setup action/observation space adapters
        self._setup_space_adapters()
        
        # Create optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup mixed precision training
        self.scaler = None
        if self.config.mixed_precision in ["fp16", "bf16"]:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.config.mixed_precision == "fp16")
            )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        
        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint is not None:
            self._resume_from_checkpoint()
    
    def _setup_model(self) -> None:
        """Setup model for action-head-only training."""
        # Load pretrained weights if specified
        if self.config.model_path is not None:
            load_pretrained_model(
                self.config.model_path,
                self.model,
                device=self.device,
                strict=False,
            )
        
        # Setup freezing
        setup_action_head_only_training(
            self.model,
            model_type=self.config.model_type,
            also_train_projections=self.config.train_projections,
        )
        
        # Print trainable parameters
        print_trainable_parameters(self.model, verbose=False)
    
    def _setup_data(self) -> None:
        """Setup data loaders."""
        # Create dataset adapter config
        adapter_config = DatasetAdapterConfig(
            repo_id=self.config.data.repo_id,
            root=self.config.data.root,
            max_action_dim=self.config.max_action_dim,
            max_state_dim=self.config.max_state_dim,
            image_keys=self.config.data.image_keys,
            resize_images=self.config.data.resize_images,
            n_obs_steps=self.config.data.n_obs_steps,
            chunk_size=self.config.data.chunk_size,
            split=self.config.data.split,
        )
        
        # Create training data loader
        self.train_loader = create_lerobot_dataloader(
            adapter_config,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=self.config.data.shuffle,
            pin_memory=self.config.data.pin_memory,
        )
        
        # Create evaluation data loader
        eval_adapter_config = DatasetAdapterConfig(
            repo_id=self.config.data.repo_id,
            root=self.config.data.root,
            max_action_dim=self.config.max_action_dim,
            max_state_dim=self.config.max_state_dim,
            image_keys=self.config.data.image_keys,
            resize_images=self.config.data.resize_images,
            n_obs_steps=self.config.data.n_obs_steps,
            chunk_size=self.config.data.chunk_size,
            split=self.config.eval_split,
        )
        
        self.eval_loader = create_lerobot_dataloader(
            eval_adapter_config,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            pin_memory=self.config.data.pin_memory,
        )
        
        # Store dataset adapter for stats
        self.dataset = LeRobotDatasetAdapter(adapter_config)
    
    def _setup_space_adapters(self) -> None:
        """Setup action and observation space adapters."""
        # Get dataset stats
        dataset_stats = self.dataset.get_stats()
        feature_info = self.dataset.get_feature_info()
        
        # Convert stats to tensors if needed
        if isinstance(dataset_stats, dict):
            for key in dataset_stats:
                if not isinstance(dataset_stats[key], torch.Tensor):
                    dataset_stats[key] = torch.tensor(dataset_stats[key])
        
        # Create action space config
        action_config = create_action_space_config_from_dataset(
            dataset_stats,
            original_action_dim=feature_info["original_action_dim"],
            max_action_dim=self.config.max_action_dim,
            use_projection=self.config.use_action_projection,
        )
        
        # Create observation space config
        obs_config = create_observation_space_config_from_dataset(
            dataset_stats,
            original_state_dim=feature_info["original_state_dim"],
            max_state_dim=self.config.max_state_dim,
            use_projection=self.config.use_state_projection,
        )
        
        # Create adapters
        self.action_adapter = ActionSpaceAdapter(action_config).to(self.device)
        self.obs_adapter = ObservationSpaceAdapter(obs_config).to(self.device)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        opt_config = self.config.optimizer
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer
        if opt_config.name.lower() == "adamw":
            self.optimizer = AdamW(
                trainable_params,
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name.lower() == "adam":
            self.optimizer = Adam(
                trainable_params,
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps,
            )
        elif opt_config.name.lower() == "sgd":
            self.optimizer = SGD(
                trainable_params,
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        sched_config = self.config.scheduler
        
        # Compute total steps if not specified
        if sched_config.total_steps is None:
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.num_epochs
            if self.config.max_steps is not None:
                total_steps = min(total_steps, self.config.max_steps)
        else:
            total_steps = sched_config.total_steps
        
        # Create scheduler
        if sched_config.name.lower() == "cosine_with_warmup":
            def lr_lambda(current_step):
                if current_step < sched_config.warmup_steps:
                    return current_step / max(1, sched_config.warmup_steps)
                progress = (current_step - sched_config.warmup_steps) / max(
                    1, total_steps - sched_config.warmup_steps
                )
                return max(
                    sched_config.min_lr / self.config.optimizer.lr,
                    0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
                )
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        elif sched_config.name.lower() == "linear_warmup":
            def lr_lambda(current_step):
                if current_step < sched_config.warmup_steps:
                    return current_step / max(1, sched_config.warmup_steps)
                return max(
                    sched_config.min_lr / self.config.optimizer.lr,
                    (total_steps - current_step) / (total_steps - sched_config.warmup_steps)
                )
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        elif sched_config.name.lower() == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma,
            )
        
        elif sched_config.name.lower() == "constant":
            self.scheduler = LambdaLR(self.optimizer, lambda _: 1.0)
        
        else:
            raise ValueError(f"Unknown scheduler: {sched_config.name}")
    
    def _resume_from_checkpoint(self) -> None:
        """Resume training from checkpoint."""
        checkpoint_path = self.config.resume_from_checkpoint
        
        if checkpoint_path is None:
            # Try to find latest checkpoint
            checkpoint_path = get_latest_checkpoint(self.config.checkpoint.save_dir)
        
        if checkpoint_path is not None:
            self.global_step, self.current_epoch, _ = load_checkpoint(
                checkpoint_path,
                self.model,
                self.optimizer,
                self.scheduler,
                device=self.device,
                strict=False,
            )
            print(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # Forward pass with mixed precision
        dtype = torch.float16 if self.config.mixed_precision == "fp16" else (
            torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32
        )
        
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision != "no", dtype=dtype):
            # Call model forward (assumes model returns loss as first output)
            outputs = self.model(batch)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                loss = outputs[0]
            elif isinstance(outputs, dict):
                loss = outputs.get("loss", outputs.get("total_loss"))
            else:
                loss = outputs
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Return metrics
        return {"loss": loss.item() * self.config.gradient_accumulation_steps}
    
    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        # Clip gradients
        if self.config.optimizer.gradient_clip_norm > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer.gradient_clip_norm,
            )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Scheduler step
        self.scheduler.step()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Extract loss
            if isinstance(outputs, tuple):
                loss = outputs[0]
            elif isinstance(outputs, dict):
                loss = outputs.get("loss", outputs.get("total_loss"))
            else:
                loss = outputs
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"eval_loss": avg_loss}
    
    def train(self) -> None:
        """Main training loop."""
        print(f"\nStarting action-head-only training...")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Output directory: {self.config.output_dir}\n")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        steps_in_epoch = 0
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Progress bar for epoch
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                total=len(self.train_loader),
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Training step
                metrics = self.training_step(batch)
                
                # Update counters
                steps_in_epoch += 1
                
                # Optimizer step (with gradient accumulation)
                if steps_in_epoch % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    })
                    
                    # Log metrics
                    if self.global_step % self.config.logging.log_every_n_steps == 0:
                        self._log_metrics(metrics)
                    
                    # Save checkpoint
                    if self.global_step % self.config.checkpoint.save_every_n_steps == 0:
                        self._save_checkpoint()
                    
                    # Evaluate
                    if self.global_step % self.config.eval_every_n_steps == 0:
                        eval_metrics = self.evaluate()
                        self._log_metrics(eval_metrics)
                        
                        # Save best model
                        if eval_metrics["eval_loss"] < self.best_loss:
                            self.best_loss = eval_metrics["eval_loss"]
                            self._save_checkpoint(is_best=True)
                    
                    # Check max steps
                    if (self.config.max_steps is not None and 
                        self.global_step >= self.config.max_steps):
                        print(f"\nReached max steps: {self.config.max_steps}")
                        return
            
            # Reset epoch counter
            steps_in_epoch = 0
            
            # Epoch complete
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} complete in {epoch_time:.2f}s")
        
        print("\nTraining complete!")
        
        # Final checkpoint
        self._save_checkpoint(is_best=False)
    
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # TODO: Add W&B logging support
        pass
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint_name = "best" if is_best else None
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            save_dir=self.config.checkpoint.save_dir,
            checkpoint_name=checkpoint_name,
            metadata={
                "best_loss": self.best_loss,
                "model_type": self.config.model_type,
            },
            save_optimizer=self.config.checkpoint.save_optimizer_state,
            save_scheduler=self.config.checkpoint.save_scheduler_state,
        )
        
        # Cleanup old checkpoints
        if not is_best:
            cleanup_old_checkpoints(
                self.config.checkpoint.save_dir,
                keep_last_n=self.config.checkpoint.keep_last_n,
            )


