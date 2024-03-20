import sys
from typing import Tuple
sys.path.append("../")

from root import DigitUnit
from root import save


digit_model = DigitUnit(784, 10)
save(digit_model)