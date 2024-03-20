import os
import json

from . import parser
DEFAULT = os.getenv("DEFAULT")

from torch.utils.data import DataLoader, random_split
from load_torch import get_cls_from_torch, Compose
from root import load_model, get_cls
from root import Learning

tmp = parser.add_argument_group(title="Nhóm lệnh dùng để huấn luyện mô hình", description="Sử dụng các cờ hiệu")
tmp.add_argument('--trainable', type=bool, help="Cờ hiệu dùng để thông báo xảy ra quá trình huấn luyện", action="store")
tmp.add_argument('--train_file', type=str, help="Đọc từ file cấu hình (các thông tin quy định quá trình đào tạo)", default=DEFAULT)

def __check_file(file):
    ext = file.split('.')[1]
    if ext != "json":
        raise ValueError("Đuôi file không được chấp nhận!")
    with open(file, "rb") as f:
        dt = json.load(f)
    return dt

def __build_learn_object(cfg):
    model = load_model(cfg["model"])
    loss = get_cls_from_torch(cfg["loss"])
    optimizer = get_cls_from_torch(cfg["optimizer"][0], 
        model.parameters() ,**cfg["optimizer"][1])
    accuracy = get_cls_from_torch(cfg["accuracy"][0], **cfg["accuracy"][1])
    
    callbacks = []
    for name, kwargs in cfg["callbacks"]:
        callbacks.append(
            get_cls(name, **kwargs)
        )

    learning = Learning(model)
    learning.set(loss, optimizer, accuracy, device=cfg["device"], callbacks=callbacks)

    return learning

def __build_training_data(cfg):
    _transform = []
    for name, kwargs in cfg["dataset"][1]["transform"]:
        _transform.append(
            get_cls_from_torch(name, **kwargs)
        )
    
    if len(_transform) > 0:
        _transform = Compose(_transform)

    dataset = get_cls_from_torch(cfg["dataset"][0], 
        root=cfg["dataset"][1]["root"], transform=_transform, **cfg["dataset"][1]["other"])
    train_dataset, val_dataset = random_split(dataset, cfg["split_size"])
    train_loader = DataLoader(train_dataset, **cfg["loader"]["train"])
    val_loader = DataLoader(val_dataset, **cfg["loader"]["valid"])
    return train_loader, val_loader

def train(args):
    if args.trainable:
        train_file = args.train_file
        if train_file == DEFAULT:
            print("Không thể huấn luyện, do không có file config")
            return
        config = __check_file(train_file)
        # if os.path.exists(config["history_storage"]):
        #     print("File dùng để lưu trữ lịch sử huấn luyện đã tồn tại!")
        #     return
        
        learning = __build_learn_object(config)
        train_loader, val_loader = __build_training_data(config)

        his = learning.learn(config["epoch"], train_loader, val_loader, show_progress=config["show_progress"])
        with open(config["history_storage"], "w", encoding="utf-8") as f:
            f.write(his)