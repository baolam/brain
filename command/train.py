import os
import json
from .. import load_model
from ..learn import Learning
from . import parser

tmp = parser.add_argument_group(title="Huấn luyện mô hình", description="Các thông số/ cờ hiệu dùng để điều khiển quá trình huấn luyện")
tmp.add_argument("--trainable", help="Cho phép huấn luyện mô hình", type=bool, default=False)
tmp.add_argument("--batch_size", help="Kích thước của một bước huấn luyện", default=64, type=int)
tmp.add_argument("--device", help="Thiết bị dùng để huấn luyện", type=str, default="cpu")
tmp.add_argument("--epoch", help="Số bước huấn luyện", type=int)
tmp.add_argument("--valid_size", help="Dùng để tách tỉ lệ dữ liệu huấn luyện", type=float, default=0.2)
tmp.add_argument("--dataset_folder", help="Thư mục lưu trữ dữ liệu huấn luyện")
tmp.add_argument("--train_file", help="File lưu trữ các thông tin huấn luyện", type=str, default=None)
tmp.add_argument("--model", help="File chứa mô hình huấn luyện", type=str)

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
    args.epoch = data["epoch"]
    args.batch_size = data["batch_size"]
    args.model = data["model"]
    args.loss = data["loss"]
    args.optimizer = data["optimizer"]
    args.valid_size = data["valid_size"]
    args.dataset_folder = data["dataset_folder"]
    args.accuracy = data["accuracy"]
    args.device = data["device"]
    args.callbacks = data["callbacks"]
    return args

def train(args):
    if args.trainable:
        tmp = args
        train_file = tmp.train_file
        if not train_file is None:
            tmp = __load_from_file(train_file, tmp)
        model = load_model(tmp.model)
        learning = Learning(model)
        learning.set(
            tmp.loss, tmp.optimizer, tmp.accuracy, tmp.device, tmp.callbacks
        )