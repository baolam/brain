import os
import json
from . import parser

tmp = parser.add_argument_group(title="Huấn luyện mô hình", description="Các thông số/ cờ hiệu dùng để điều khiển quá trình huấn luyện")
tmp.add_argument("--trainable", help="Cho phép huấn luyện mô hình", type=bool, default=False)
tmp.add_argument("--batch_size", help="Kích thước của một bước huấn luyện", default=64, type=int)
tmp.add_argument("--device", help="Thiết bị dùng để huấn luyện", type=str, default="cpu")
tmp.add_argument("--epoch", help="Số bước huấn luyện", type=int)
tmp.add_argument("--dataset_folder", help="Thư mục lưu trữ dữ liệu huấn luyện")
tmp.add_argument("--train_file", help="File lưu trữ các thông tin huấn luyện", type=str, default=None)

# Cài đặt huấn luyện từ một file
# Cấu trúc một lệnh train
# Bộ dataset?
# Split ra sao?
# Batch_size?
# Hàm chọn độ chính xác?

def __load_from_file(file, args):
    ext = file.split('.')[1]
    if ext != "json":
        raise ValueError("Dạng file mở rộng không được chấp nhận!")
    data = json.load(file)
    print(data)

def train(args):
    if args.trainable:
        train_file = args.train_file
        if not train_file is None:
            __load_from_file(train_file, args)