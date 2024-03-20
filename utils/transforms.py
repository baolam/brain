from torchvision.transforms import ToTensor, Compose

default_transform = Compose([
    ToTensor()
])