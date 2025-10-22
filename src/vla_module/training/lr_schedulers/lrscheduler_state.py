from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from ..utils import flatten_dict, unflatten_dict, write_json
from ..utils import deserialize_json_into_object


SCHEDULER_STATE="scheduler_state.json"

def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
