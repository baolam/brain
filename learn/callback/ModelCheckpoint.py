from typing import Any
from torch import inf

from .Callback import Callback

class ModelCheckpoint(Callback):
    def __init__(self, storage : str):
        super().__init__()
        self._best_loss = inf
        self.storage = storage

    def __call__(self, *args: Any, **kwds: Any):
        epoch = kwds.get("epoch")
        tl, ta, vl, va = args
