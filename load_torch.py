import torch
import torchmetrics

from torch.nn import *
from torch.optim import *
from torchmetrics import *
from torchvision.datasets import *

import importlib
m_torch = importlib.import_module("load_torch")

def get_cls_from_torch(cls_name, *args, **kwargs):
    '''
    Hàm trả về bộ dựng lớp của thư viện
    '''
    return getattr(m_torch, cls_name)(*args, **kwargs)