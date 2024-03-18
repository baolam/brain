import sys
sys.path.append("../../")

import brain
bceloss = brain.get_cls("BCELoss")
adam = brain.get_cls("Adam")