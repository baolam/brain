from abc import ABC, abstractmethod
from typing import Tuple, Dict

from torch import Tensor, stack, concatenate
from . import Unit


LAYER = "combine_unit"
class CombineUnit(Unit, ABC):
    def __init__(self, addr: Tuple[str, None], **kwargs):
        super().__init__(addr, LAYER, **kwargs)

    def send(self, *args, **kwargs):
        pass


class ConcatFeature(CombineUnit):
    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, **kwargs)
        self.__feature : Dict[str, Tensor] = {  }
        self.__infor : Dict[str, str] = {  }

    def recv(self, x: Tensor, idx : int = None, _from : str = None, *args, **kwargs):
        # Làm sao quyết định vị trí của đặc trưng?
        if idx is None:
            raise ValueError("Không có chỉ số khoảng!")
        if _from is None:
            raise ValueError("Đơn vị không tồn tại tên")
        if not self.__feature.get(str(idx)) is None:
            raise ValueError("Đã tồn tại!")
        
        self.__feature[str(idx)] = x
        self.__infor[str(idx)] = _from

    def forward(self, *args, **kwargs) -> Tensor:
        return self.feature()
    
    def feature(self, *args, **kwargs):
        output = []
        for key in sorted(self.__feature.keys()):
            output.append(self.__feature[key])
        return concatenate(output)

    def clear_feature(self, *args, **kwargs):
        self.__feature.clear()

    def infor_feature(self):
        return self.__infor


class MeanFeatureWithoutAdaptive(CombineUnit):
    # Tổng hợp đặc trưng với tỉ lệ đóng góp như nhau

    def __init__(self, addr: Tuple[str | None], **kwargs):
        super().__init__(addr, **kwargs)
        self.__feature = []

    def forward(self, *args ,**kwargs):
        x = self.feature()
        x = x.mean(dim = 1)
        return x

    def recv(self, x: Tensor, _from: str = None, *args, **kwargs):
        self.__feature.append(x)
    
    def clear_feature(self):
        self.__feature.clear()

    def feature(self):
        # Tiến hành cài đặt chuyển đổi
        return stack(self.__feature)        
