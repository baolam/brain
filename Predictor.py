from typing import Tuple, Tensor
from graph.ForwardGraph import ForwardGraph
from graph.ForGraphMulInputs import ForGraphMulInputs
from root import load_model


class Predictor():
    def __init__(self, model : Tuple[str | ForwardGraph | ForGraphMulInputs]):
        self.model = model
        if isinstance(model, str):
            self.model = load_model(model)
        
    def cluster_input(self, x : Tensor, _from):
        if isinstance(self.model, ForwardGraph):
            return
        if not isinstance(x, Tensor):
            raise ValueError("Đầu vào đơn vị ko đúng định dạng")
        self.model.recv_input(x, _from)

    def clear_input(self):
        if isinstance(self.model, ForwardGraph):
            return
        self.model.clear_input()
    
    def forward(self, x = None ,*args, **kwargs):
        return self.model.forward_no_grad(x)