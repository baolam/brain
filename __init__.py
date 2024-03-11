import torch

from .utils import *
from .visitor import *
from .units import *
from .graph import *
from .learn import *

import importlib
brain = importlib.import_module("brain")

def get_cls(cls_name):
    '''
    Hàm trả về bộ dựng lớp của thư viện
    '''
    return getattr(brain, cls_name)