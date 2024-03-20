import json
import atexit

from root import *

# Cấu trúc tải thông tin liên quan đến phiên làm việc
_infor_file = "E:/my_research/brain/config/infor.json"

dt = json.load(_infor_file)
set_counter(dt["counter"])

def __session_end():
    global dt, _infor_file
    dt["counter"] = get_counter()
    json.dump(dt, _infor_file)

atexit.register(__session_end)