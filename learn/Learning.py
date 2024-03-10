from torch import nn
from torch import optim
from .LearnGraph import LearnGraph


class Learning():
    def __init__(self, target : LearnGraph):
        self._target = target
        self._run_at = None
        self._optimizer = None
        self._accuracy = None
        self._loss = None

    def set_optimizer(self, optimizer : optim.Optimizer, *args, **kwargs):
        self._optimizer = optimizer(self._target.parameters(), *args, **kwargs)
    
    def set_device(self, _device : str = "cpu"):
        self._run_at = _device
        self._target = self._target.to(_device)

    def set(self, loss, optimizer, accuracy = None, 
        device : str = "device", *args ,**kwargs):
        self._loss = loss
        self._accuracy = accuracy
        self.set_optimizer(optimizer, *args, **kwargs)
        self.set_device(device)
    
    def train(self):
        pass

    def valid(self):
        pass

    def test(self):
        pass

    def __optimize(self):
        pass

    def show_epoch(self):
        pass

    def show_infor(self):
        print("Thông tin cơ bản về đào tạo mô hình")
        print("-----------------------------------")
        print("Quá trình đào tạo được thực hiện trên thiết bị: {}".format(self._run_at))
        print("Tổng số thông số của mô hình là:                {}".format(self._target.total_params()))
        print("Tổng số thông số huấn luyện là:                 {}".format(self._target.train_params()))
        print("Tối ưu dựa trên thuật toán : {}".format(str(self._optimizer)))
        print("Hàm lỗi :                    {}".format(str(self._loss)))

    def learn(self):
        pass