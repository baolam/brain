from typing import Tuple
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor


class Unit(ABC, nn.Module):
    def __init__(self, addr : Tuple[str, None], layer : Tuple[str, None], **kwargs):
        '''
            Hàm khởi tạo của một Unit.
            Kế thừa từ lớp trừu tượng và lớp nn.Module thể hiện chức năng quản lí và đóng vai trò như một
            đơn vị tính toán trong mô hình.

            Args:
            addr : địa chỉ của unit (giống số nhà)
            layer : phân lớp của unit (dùng để nhóm các unit lại hỗ trợ cho quá trình lan truyền, ...)
        '''
        super().__init__()
        # Địa chỉ của unit (xem như tên)
        self.__address = addr
        # Phân lớp của unit
        self.__layer = layer
        # Khả năng học của đơn vị
        self.__learnable = True

    def name(self):
        return self.__address
    
    def layer(self):
        return self.__layer
    
    def set_layer(self, layer):
        self.__layer = layer
    
    def learnable(self):
        return self.__learnable
    
    def set_learn(self, learnable : bool):
        if self.learnable() == learnable:
            return
        self = self.requires_grad_(learnable)
        self.__learnable = learnable

    def show_infor(self):
        learn_msg = "không có khả năng tối ưu"
        if self.learnable():
            learn_msg = "có khả năng tối ưu"
        print('Đơn vị {} ở phân lớp {} {}'.format(self.name(), self.layer(), learn_msg))
    
    def __params(self, trainable : bool = False):
        if trainable:
            return sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        return sum(
            p.numel() for p in self.parameters()
        )

    def total_params(self):
        return self.__params()
    
    def train_params(self):
        return self.__params(trainable=True)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        pass
    
    @abstractmethod
    def recv(self, x : Tensor, _from : str = None):
        pass

    @abstractmethod
    def feature(self):
        '''
        Trả về đặc trưng đầu vào của đơn vị
        '''
        pass
    
    @abstractmethod
    def send(self):
        pass
    
    @abstractmethod
    def clear_feature(self):
        pass