from typing import Tuple, List

from torch import Tensor, stack, tensor, argmax
from torch import nn
from . import Unit


class MemoryUnit(Unit):
    LAYER = "memory"
    def __init__(self, addr: Tuple[str, None], **kwargs):
        super().__init__(addr, self.LAYER, **kwargs)
        self.__express : List[Tensor] = []
        self.__addr : List[str] = []
        self.__inp = tensor()
        self.activate = nn.Softmax(dim = 1)

    def recv(self, x: Tensor, _from: str = None, *args, **kwargs):
        self.__addr.append(_from)
        self.__express.append(x)
        self.__inp = stack(self.__addr)

    def feature(self, *args, **kwargs):
        return self.__inp
    
    def clear_feature(self, *args, **kwargs):
        pass

    def send(self, *args, **kwargs):
        pass

    def forward(self, x : Tensor ,*args, **kwargs) -> Tensor:
        a = x @ self.feature()
        a = self.activate(a)
        return a
    
    def belongs(self, p : Tensor) -> List[Tensor]:
        '''
        Trả về địa chỉ đơn vị tuân theo xác suất 
        '''
        out = []
        a = argmax(p, dim = 1)
        for i in range(a.size()[0]):
            out.append(
                self.__express[a[i]]
            )

        return out