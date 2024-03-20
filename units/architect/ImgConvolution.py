from torch import Tensor, nn
from typing import Tuple
from ..RepresentUnit import AutoRepresent
from utils.address import get


class DefaultImgConv(AutoRepresent):
    def __init__(self, channels : int, flatten_dim : int, output_dim : int, addr: Tuple[str, None] = None, **kwargs):
        if addr is None:
            addr = get()
        super().__init__(addr, **kwargs)
        self.handle = nn.Sequential(
            nn.Conv2d(channels, 2, (3, 3)),
            nn.Conv2d(2, 2, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Flatten()
        )

        self.represent = nn.Sequential(
            nn.Linear(flatten_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x ,*args, **kwargs) -> Tensor:
        x = self.handle(x)
        x = self.represent(x)
        return x