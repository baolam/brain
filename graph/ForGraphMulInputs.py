from typing import List, Tuple, Dict
from torch import Tensor, tensor
from units.Unit import Unit
from units.Dictionary import Dictionary
from units.RepresentUnit import AutoRepresent
from .ForwardGraph import ForwardGraph


class ForGraphMulInputs(ForwardGraph):
    def __init__(self, dictionary: Dictionary = None, 
        edges: List[Tuple[Tuple[str, Tuple[int, Unit]], Tuple[str, Tuple[int, Unit]]]] = None):
        super().__init__(dictionary, edges)
        self.__represents : Dict[str, Tensor] = { }
        self.__init_represent()

    def __init_represent(self):
        for represent in self.units_by_layer(AutoRepresent.LAYER, return_addr=True):
            self.__represents[represent] = tensor()
    
    def recv_input(self, x : Tensor, _from : str, *args, **kwargs):
        self.__represents[_from] = x

    def clear_input(self):
        self.__represents.clear()
        self.__init_represent()

    def forward(self, *args, **kwargs):
        pass