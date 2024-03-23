from typing import Tuple
from graph.ForwardGraph import ForwardGraph
from graph.ForGraphMulInputs import ForGraphMulInputs
from root import load_model


class Predictor():
    def __init__(self, model : Tuple[str | ForwardGraph | ForGraphMulInputs]):
        self.model = model
        if isinstance(model, str):
            self.model = load_model(model)
    
    def cluster_input(self):
        if isinstance(self.model, ForwardGraph):
            return
    
    def clear_input(self):
        if isinstance(self.model, ForwardGraph):
            return
    