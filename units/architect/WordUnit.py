from typing import Tuple
from .. import Unit


class Word(Unit):
    def __init__(self, addr: Tuple[str, None] = None, **kwargs):
        if addr is None:
            raise ValueError("Đơn vị từ không thể None")
        super().__init__(addr, **kwargs)