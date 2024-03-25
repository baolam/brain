from typing import Tuple

from torch import Tensor
from torch import nn
from .. import Unit
from ...root import get

class DigitUnit(Unit):
    def __init__(self, inp_dim : int, out_dim : int, 
        addr: Tuple[str, None] = None, layer: Tuple[str, None] = "Digit_Unit", **kwargs):
        if addr is None:
            addr = get()
        super().__init__(addr, layer, **kwargs)
        self.ff = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Softmax(dim = 1)
        )
    
    def send(self, *args, **kwargs):
        pass
    
    def recv(self, x: Tensor, _from: str = None, *args, **kwargs):
        pass

    def feature(self, *args, **kwargs):
        pass

    def clear_feature(self, *args, **kwargs):
        pass

    def forward(self, x : Tensor,*args, **kwargs) -> Tensor:
        x = x.reshape((x.size()[0], -1))
        return self.ff(x)