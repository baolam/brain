from abc import ABC, abstractmethod
from typing import Tuple, Dict

from torch import Tensor, stack
from . import Unit


LAYER = "combine_unit"
class CombineUnit(Unit, ABC):
    def __init__(self, addr: Tuple[str, None], **kwargs):
        super().__init__(addr, LAYER, **kwargs)

    @abstractmethod
    def forward(self):
        pass

    def send(self):
        pass


class ConcatFeature(CombineUnit):
    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, **kwargs)
        self.__feature : Dict[str, Tensor] = {  }

    def recv(self, x: Tensor, _from: str = None):
        # Làm sao quyết định vị trí của đặc trưng?
        if _from is None:
            raise ValueError("Không chỉ rõ đơn vị đầu vào!")
        if not self.__feature.get(_from) is None:
            raise ValueError("Đặc trưng đã tồn tại!")
        self.__feature[_from] = x

    def forward(self):
        pass

    def feature(self):
        pass

    def clear_feature(self):
        self.__feature.clear()


class MeanFeatureWithoutAdaptive(CombineUnit):
    # Tổng hợp đặc trưng với tỉ lệ đóng góp như nhau

    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, **kwargs)
        self.__feature = []

    def forward(self, *args ,**kwargs):
        x = self.feature()
        x = x.mean(dim = 1)
        return x

    def recv(self, x: Tensor, _from: str = None):
        self.__feature.append(x)
    
    def clear_feature(self):
        self.__feature.clear()

    def feature(self):
        # Tiến hành cài đặt chuyển đổi
        return stack(self.__feature)        
