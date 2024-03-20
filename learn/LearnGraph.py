from typing import List, Tuple
from torch import nn, no_grad

from graph.Graph import Graph
from units.Dictionary import Dictionary
from units.Unit import Unit
from graph.ForwardGraph import ForwardGraph


class LearnGraph(ForwardGraph, nn.Module):
    def __init__(self, dictionary: Dictionary = None, 
        edges: List[Tuple[Tuple[str, Tuple[int, Unit]], Tuple[str, Tuple[int, Unit]]]] = None):
        super().__init__(dictionary, edges)
    
    def total_params(self):
        params = 0
        for unit in self._units.alls():
            params += unit.total_params()
        return params
    
    def train_params(self):
        params = 0
        for unit in self._units.units():
            params += unit.train_params()
        return params
    

    def learnable(self):
        learnable = {  }
        for unit in self._units.alls():
            learnable[unit.name()] = self._learnable(unit.learnable())
        return learnable
    
    def forward_no_grad(self, x):
        with no_grad():
            return self.forward(x)