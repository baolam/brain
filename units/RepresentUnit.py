from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor
from . import Unit


class AutoRepresent(Unit):
    LAYER = "represent"
    # Biểu diễn tự động (tự tối ưu)

    def __init__(self, addr: Tuple[str, None], **kwargs):
        super().__init__(addr, self.LAYER, **kwargs)

    def send(self, *args, **kwargs):
        pass

    def recv(self, x: Tensor, _from: str = None, *args, **kwargs):
        pass

    def feature(self, *args, **kwargs):
        pass

    def clear_feature(self, *args, **kwargs):
        pass

    def set_layer(self, layer):
        pass


class CodingRepresent(Unit, ABC):
    LAYER = "represent"
    # Biểu diễn dựa vào đặc trưng được trích xuất thủ công

    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, self.LAYER, **kwargs)

    @abstractmethod
    def extractor(self, x):
        '''
        Bộ phận hình thành nên đặc trưng (background substractor, ...)
        '''
        pass

    def set_layer(self, layer):
        pass