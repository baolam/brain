import os
from . import parser
from . import STORAGE, VERSION

tmp = parser.add_argument_group(title="Quản lí các thông tin chung của dự án")
tmp.add_argument("-v", "--version", help="Phiên bản của brain", action="store_true")
tmp.add_argument("-storage", "--storage", help="Thư mục lưu trữ", action="store_true")

def infor(args):
    if args.version:
        print(VERSION)
    elif args.storage:
        print(STORAGE)