import os
from . import parser
from . import STORAGE, VERSION

infor = parser.add_mutually_exclusive_group(required = True)
infor.add_argument("-v", "--version", help="Phiên bản của brain", action="store_true")
infor.add_argument("-storage", "--storage", help="Thư mục lưu trữ", action="store_true")

def infor(args):
    if args.version:
        print(VERSION)
    elif args.storage:
        print(STORAGE)