import json
import atexit

from .root import *

# Cấu trúc tải thông tin liên quan đến phiên làm việc
_infor_file = "E:/my_research/brain/config/infor.json"

with open(_infor_file, "rb") as f:
    dt = json.load(f)
set_counter(dt["counter"])

def __session_end():
    global dt, _infor_file
    dt["counter"] = get_counter()
    with open(_infor_file, "w", encoding="utf-8") as f:
        json.dump(dt, f)

atexit.register(__session_end)

import torch
import torchmetrics

from torch.nn import *
from torch.optim import *
from torchmetrics import *
from torchvision.datasets import *
from torchvision.transforms import *