from abc import ABC, abstractmethod
from typing import Tuple
from . import Unit

LAYER = "represent"

class AutoRepresent(Unit, ABC):
    # Biểu diễn tự động (tự tối ưu)

    def __init__(self, addr: Tuple[str, None], **kwargs):
        super().__init__(addr, LAYER, **kwargs)


class CodingRepresent(Unit, ABC):
    # Biểu diễn dựa vào đặc trưng được trích xuất

    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, LAYER, **kwargs)

    @abstractmethod
    def extractor(self, x):
        '''
        Bộ phận hình thành nên đặc trưng (background substractor, ...)
        '''
        pass