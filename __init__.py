import torch
import torchmetrics

import json
from uuid import uuid4
from torch import load, save
from typing import Tuple

from torch.nn import *
from torch.optim import *
from torchmetrics import Accuracy
from .command import S_UNIT, S_MODEL, DIM, S_MANAGE
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

def build_unit(cls_name, *args, **kwargs):
    cls = get_cls(cls_name)
    if not isinstance(cls, Unit):
        raise ValueError("Không phải lớp kế thừa Unit")
    unit = cls(dim = DIM, *args, **kwargs)
    return unit

def __save_unit(unit : Unit):
    save(S_UNIT + '/' + unit.name() + '.pt', unit)

def __save_graph(graph : Graph, name : str):
    edges = graph.edges()
    cls = graph.__class__.__name__
    infor = {
        "edges" : edges,
        "class_name" : cls
    }
    json.dump(infor, S_MANAGE + '/' + name + '.json')    

def __save_model(model : Graph, name : str):
    if name is None:
        name = str(uuid4())
    save(S_MODEL + '/' + name + '.pt', model)
    for unit_addr in model._units.address():
        __save_unit(model._units.get(unit_addr))
    __save_graph(model, name)

def save(obj : Tuple[Unit, Graph], name : str = None):
    if isinstance(obj, Unit):
        __save_unit(obj)
    elif isinstance(obj, Graph):
        __save_model(obj, name)
    else:
        raise ValueError("Không thể lưu trữ!")

def load_model(model : str) -> torch.nn.Module:
    _model = load(S_MODEL + '/' + model)
    if not isinstance(_model, Graph):
        raise ValueError("Không phải model")
    return _model