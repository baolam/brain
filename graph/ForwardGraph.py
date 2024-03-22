from typing import List, Tuple
from torch import nn, stack, Tensor, no_grad

from units.Unit import Unit
from units.Dictionary import Dictionary
from visitor.Topo import Topo
from .Graph import Graph


class ForwardGraph(Graph):
    def __init__(self, dictionary: Dictionary = None, 
        edges : List[Tuple[Graph.UNIT_TYPE, Graph.UNIT_TYPE]] = None):
        super().__init__(dictionary)
        self.__lock_initalize = False
        self.__order : List[Unit] = nn.ModuleList()
        self.__output_size = 0
        if not edges is None:
            self.initalize(edges)

    def __build_topo(self):
        temp = Topo(self._edges)
        temp.visit()
        
        for addr in temp.topo():
            self.__order.append(self._units.get(addr))

        return temp.topo()

    def _outputs(self) -> List[str]:
        # Thông tin về đơn vị đầu ra
        # Dựa vào số bậc của đơn vị
        outputs = []
        for addr, neighbor in self._edges.items():
            if len(neighbor.content()) == 0:
                outputs.append(addr)
        return outputs

    def initalize(self, edges = None):
        if self.__lock_initalize:
            raise ValueError("Chức năng này đã bị khóa!")
        
        if not self.__lock_initalize:
            self.__lock_initalize = True
            
        if not edges is None:
            for n1, n2 in edges:
                self.add_edge(n1, n2)

        topo = self.__build_topo()
        outputs = self._outputs()
        self.__output_size = len(outputs)

        return topo, outputs

    def forward(self, x):
        if len(self.__order) == 0:
            raise ValueError("Không thể tiến hành lan truyền!")
        
        # Quá trình lan truyền gồm 2 bước:
        # Bước 1: Tính toán tại đơn vị
        # Bước 2: Lan truyền kết quả tính toán
        # Có thể cài đặt bằng phối hợp process và xem xét mỗi đơn vị
        # như một sinh vật sống - Threading

        output = []
        for unit in self.__order:
            tmp = unit(x)
            neighbor = self.neighbor(unit.name())

            if len(neighbor) == 0:
                if self.__output_size == 1:
                    return tmp
                else:
                    output.append(tmp)
            else:
                for neigh in neighbor:
                    self._units.get(neigh).recv(tmp)

        return stack(output).transpose(0, 1)
    
    def forward_no_grad(self, x):
        with no_grad():
            return self.forward(x)
        
    def __call__(self, x) -> Tensor:
        return self.forward(x)
    
    def total_params(self):
        params = 0
        for unit in self._units.alls():
            params += unit.total_params()
        return params
    
    def train_params(self):
        params = 0
        for unit in self._units.alls():
            params += unit.train_params()
        return params
    

    def learnable(self):
        learnable = {  }
        for unit in self._units.alls():
            learnable[unit.name()] = unit.learnable()
        return learnable