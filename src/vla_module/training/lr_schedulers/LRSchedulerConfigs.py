import abc
import logging
import math
from dataclasses import asdict, dataclass

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from vla_module.utils.io_utils import deserialize_json_into_object, write_json

SCHEDULER_STATE="scheduler_state.json"

@dataclass
class LRSchedulerBase(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError

@LRSchedulerBase.register_subclass("step_lr")
@dataclass
class StepLRSchedulerConfig(LRSchedulerBase):
    """Step learning rate decay scheduler.
    
    Decays the learning rate by gamma every step_size steps.
    Commonly used for simple, predictable learning rate decay.
    """
    
    num_warmup_steps: int
    step_size: int  # Period of learning rate decay
    gamma: float = 0.1  # Multiplicative factor of learning rate decay
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.num_warmup_steps:
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1
            
            # Step decay phase
            steps_after_warmup = current_step - self.num_warmup_steps
            decay_factor = self.gamma ** (steps_after_warmup // self.step_size)
            return decay_factor
        
        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerBase.register_subclass("cosine_annealing")
@dataclass
class CosineAnnealingLRSchedulerConfig(LRSchedulerBase):
    """Cosine annealing learning rate scheduler.
    
    Sets the learning rate using a cosine annealing schedule, where the learning rate
    decreases from initial lr to eta_min following a cosine curve over T_max steps.
    """
    
    num_warmup_steps: int
    T_max: int  # Maximum number of iterations for cosine annealing
    eta_min: float = 0.0  # Minimum learning rate (as fraction of peak_lr)
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale if needed
        actual_warmup_steps = self.num_warmup_steps
        actual_T_max = self.T_max
        
        if num_training_steps < (self.num_warmup_steps + self.T_max):
            available_steps = num_training_steps - self.num_warmup_steps
            if available_steps > 0:
                scale_factor = available_steps / self.T_max
                actual_T_max = available_steps
                
                logging.info(
                    f"Auto-scaling CosineAnnealing scheduler: "
                    f"num_training_steps ({num_training_steps}) < warmup + T_max. "
                    f"Scaling T_max: {self.T_max} → {actual_T_max}"
                )
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < actual_warmup_steps:
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
            
            # Cosine annealing phase
            steps_since_warmup = current_step - actual_warmup_steps
            steps_since_warmup = min(steps_since_warmup, actual_T_max)
            
            # Cosine annealing formula: eta_min + (1 - eta_min) * (1 + cos(pi * t / T_max)) / 2
            cosine_factor = 0.5 * (1 + math.cos(math.pi * steps_since_warmup / actual_T_max))
            return self.eta_min + (1 - self.eta_min) * cosine_factor
        
        return LambdaLR(optimizer, lr_lambda, -1)

@LRSchedulerBase.register_subclass("multi_step_lr")
@dataclass
class MultiStepLRSchedulerConfig(LRSchedulerBase):
    """Multi-step learning rate decay scheduler.
    
    Decays the learning rate by gamma at specified milestone steps.
    Useful when you know specific points where LR should drop.
    """
    
    num_warmup_steps: int
    milestones: list[int]  # List of step indices where LR drops
    gamma: float = 0.1
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Sort milestones for efficient lookup
        sorted_milestones = sorted(self.milestones)
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.num_warmup_steps:
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1
            
            # Multi-step decay phase
            steps_after_warmup = current_step - self.num_warmup_steps
            decay_count = sum(1 for m in sorted_milestones if steps_after_warmup >= m)
            return self.gamma ** decay_count
        
        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerBase.register_subclass("linear_decay")
@dataclass
class LinearDecaySchedulerConfig(LRSchedulerBase):
    """Linear decay learning rate scheduler.
    
    Linearly decays the learning rate from 1.0 to end_factor over num_decay_steps.
    Simple and predictable decay pattern.
    """
    
    num_warmup_steps: int
    num_decay_steps: int
    end_factor: float = 0.0  # Final LR multiplier
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps
        
        if num_training_steps < (self.num_warmup_steps + self.num_decay_steps):
            available_steps = num_training_steps - self.num_warmup_steps
            if available_steps > 0:
                actual_decay_steps = available_steps
                
                logging.info(
                    f"Auto-scaling LinearDecay scheduler: "
                    f"Scaling decay_steps: {self.num_decay_steps} → {actual_decay_steps}"
                )
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < actual_warmup_steps:
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
            
            # Linear decay phase
            steps_since_warmup = current_step - actual_warmup_steps
            if steps_since_warmup >= actual_decay_steps:
                return self.end_factor
            
            decay_progress = steps_since_warmup / actual_decay_steps
            return 1.0 - (1.0 - self.end_factor) * decay_progress
        
        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerBase.register_subclass("exponential_decay")
@dataclass
class ExponentialDecaySchedulerConfig(LRSchedulerBase):
    """Exponential decay learning rate scheduler.
    
    Decays the learning rate exponentially by gamma every step.
    Provides smooth, continuous decay.
    """
    
    num_warmup_steps: int
    gamma: float = 0.99  # Decay rate per step
    
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.num_warmup_steps:
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1
            
            # Exponential decay phase
            steps_after_warmup = current_step - self.num_warmup_steps
            return self.gamma ** steps_after_warmup
        
        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerBase.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerBase):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


