import sys
from typing import Tuple
sys.path.append("../")

from root import DigitUnit
from root import ForwardGraph
from root import Dictionary
from root import save


digit_model = DigitUnit(784, 10)
dic = Dictionary()
dic.add(digit_model)
fg = ForwardGraph(dic)
fg.initalize()

save(fg, "digit")