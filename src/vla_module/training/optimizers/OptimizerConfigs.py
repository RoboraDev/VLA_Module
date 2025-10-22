import abc
import torch
import draccus
from dataclasses import asdict, dataclass, field


# BaseOptimizer definition of data class
# Supported optimizers for now:- SGD/SGD_With_Momentum, Adam, AdamW.
# using draccus will help me to later simplify the optimizer selection via YAML or cmd argument passing
@dataclass
class BaseOptimizer(draccus.ChoiceRegistry, abc.ABC):
    lr: float
    weight_decay: float
    grad_clip_norm: float

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @classmethod
    def default_choice_name(cls) -> str:
        return "adam"

    @abc.abstractmethod
    def build(self) -> torch.optim.Optimizer:
        """
        Build the optimizer with the provided parameters & hyper parameters configurations.

        Returns:
            The single optimizer with params and hyper-params initialized.
        """
        raise NotImplementedError
    

@BaseOptimizer.register_subclass("sgd")
@dataclass
class SGDConfig(BaseOptimizer):
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.SGD(params, **kwargs)

# Default optimizer when specific one is not provided
@BaseOptimizer.register_subclass("adam")
@dataclass
class AdamConfig(BaseOptimizer):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)           # beta1, beta2 respectively
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.Adam(params, **kwargs)


@BaseOptimizer.register_subclass("adamw")
@dataclass
class AdamWConfig(BaseOptimizer):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)           # beta1, beta2 respectively
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        kwargs.pop("grad_clip_norm")
        return torch.optim.AdamW(params, **kwargs)


