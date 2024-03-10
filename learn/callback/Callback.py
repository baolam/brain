from abc import ABC, abstractmethod
from typing import Any


class Callback(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any):
        pass