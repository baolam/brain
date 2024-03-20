from typing import Tuple
from torch import nn

from .. import Unit
from utils.address import get

class FFNReasoning(Unit):
    def __init__(self, addr: Tuple[str, None], inp_dim : int, output_dim : int, layer: Tuple[str, None] = "FFN_REASONING", **kwargs):
        if addr is None:
            addr = get()

        super().__init__(addr, layer, **kwargs)
