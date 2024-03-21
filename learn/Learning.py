import os

from typing import List, Tuple
from tqdm import tqdm
from torchmetrics import Accuracy

from torch.utils.data import DataLoader
from torch import optim

from .callback.Callback import Callback
from .LearnGraph import LearnGraph


class Learning():
    def __init__(self, target : LearnGraph):
        self._target = target
        self._run_at = None
        self._optimizer : optim.Optimizer = None
        self._accuracy = None
        self._loss = None
        # Các hàm được gọi trong quá trình tối ưu (chẳng hạn như checkpoint ...)
        # Quá trình phụ
        self._callbacks : List[Callback] = []

    def set_optimizer(self, optimizer : optim.Optimizer, *args, **kwargs):
        # self._optimizer = optimizer(self._target.parameters(), *args, **kwargs)
        self._optimizer = optimizer
    
    def set_device(self, _device : str = "cpu"):
        self._run_at = _device
        self._target = self._target.to(_device)

    def set(self, loss, optimizer, accuracy : Accuracy = None, 
        device : str = "device", callbacks : List[Callback] = [], *args ,**kwargs):
        if not accuracy is None:
            accuracy = accuracy.to(device)
        self._accuracy = accuracy
        self._loss = loss
        self.set_optimizer(optimizer, *args, **kwargs)
        self.set_device(device)

        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError("Hàm gọi lại không phải lớp con của Callback")
        self._callbacks = callbacks

    def __optimize(self, y, y_hat):
        self._optimizer.zero_grad()
        l = self._loss(y_hat, y)
        l.backward()
        self._optimizer.step()
        return l.item()

    def train(self, train : DataLoader, show_progress : bool = False):
        loss = 0.
        acc = 0.

        if show_progress:
            train = tqdm(train)

        for x, y in train:
            x, y = x.to(self._run_at), y.to(self._run_at)
            y_hat = self._target.forward(x)
            loss += self.__optimize(y, y_hat)

            if not self._accuracy is None:
                acc += self._accuracy(y_hat, y)

        return loss / len(train), acc / len(train)

    def valid(self, val : DataLoader, show_progress : bool = False):
        loss = 0.
        acc = 0.

        if show_progress:
            val = tqdm(val)
        for x, y in val:
            x, y = x.to(self._run_at), y.to(self._run_at)
            y_hat = self._target.forward_no_grad(x)
            loss += self._loss(y_hat, y).item()

            if not self._accuracy is None:
                acc += self._accuracy(y_hat, y)

        return loss / len(val), acc / len(val)
    
    def test(self, test : DataLoader):
        self._target = self._target.eval()
        return self.valid(test)

    def show_epoch(self, e : int, train_infor : Tuple[float, float], val_infor : Tuple[float, float] = None):
        print("Epoch : {}. Train_loss : {}. Train_acc : {}. Val_loss : {}. Val_acc : {}".format(
            e, train_infor[0], train_infor[1], val_infor[0], val_infor[1]
        ))

    def learn(self, epochs : int, train : DataLoader, 
        val : DataLoader = None, show_progress : bool = True):
        self.show_infor()

        infor = []

        self._target = self._target.train()
        for e in range(1, epochs + 1):
            train_loss, train_acc = self.train(train, show_progress)
            val_loss, val_acc = os.getenv("NOT_EXIST"), os.getenv("NOT_EXIST")
            if not val is None:
                val_loss, val_acc = self.valid(val)
            if show_progress:
                self.show_epoch(e, (train_loss, train_acc), (val_loss, val_acc))
            _tmp = (train_loss, train_acc, val_loss, val_acc)
            # Hàm gọi lại cho lưu trữ thông tin bổ sung
            for callback in self._callbacks:
                callback(*_tmp, epoch=e, target=self._target)
            infor.append(_tmp)
        self._target = self._target.eval()

        return infor

    def show_infor(self):
        print("Thông tin cơ bản về đào tạo mô hình")
        print("-----------------------------------")
        print("Quá trình đào tạo được thực hiện trên thiết bị: {}".format(self._run_at))
        print("Tổng số thông số của mô hình là: {}".format(self._target.total_params()))
        print("Tổng số thông số huấn luyện là: {}".format(self._target.train_params()))
        print("Tối ưu dựa trên thuật toán : {}".format(str(self._optimizer.__class__.__name__)))
        print("Hàm lỗi : {}".format(str(self._loss.__class__.__name__)))
