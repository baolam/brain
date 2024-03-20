import sys
sys.path.append("../")

import load_torch

bceloss = load_torch.get_cls_from_torch("BCELoss")
adam = load_torch.get_cls_from_torch("Adam")