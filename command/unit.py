import os

from . import parser
from . import S_UNIT

tmp = parser.add_argument_group(title="Quản lí các đơn vị lưu trữ")
tmp.add_argument("-units", '--u', help="Số lượng đơn vị hiện có!", action="store_true")


def unit(args):
    pass