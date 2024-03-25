from . import *

import importlib
m_torch = importlib.import_module("brain")

def get_cls_from_torch(cls_name ,*args, **kwargs):
    '''
    Hàm trả về bộ dựng lớp của thư viện
    '''
    return getattr(m_torch, cls_name)(*args, **kwargs)