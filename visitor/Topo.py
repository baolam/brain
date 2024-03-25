from typing import Dict
from ..utils.array import NoDuplicateArray


class Topo:
    def __init__(self, edges : Dict[str, NoDuplicateArray]):
        self.__edges = edges
        self.__visited : Dict[str, bool] = { }
        self.__topo = []

    def initalize(self):
        for edge, __ in self.__edges.items():
            self.__visited[edge] = False

    def __dfs(self, addr):
        self.__visited[addr] = True

        for neigh in self.__edges[addr].content():
            if not self.__visited[neigh]:
                self.__dfs(neigh)

        self.__topo.append(addr)

    def visit(self):
        self.initalize()
        for addr in self.__edges.keys():
            if not self.__visited[addr]:
                self.__dfs(addr)
        self.__topo.reverse()
        if self._cyclic():
            raise ValueError("Đồ thị tồn tại chu trình! Không thể tạo thứ tự topo")

    def _cyclic(self):
        pos = { }
        for i in range(len(self.__topo)):
            pos[self.__topo[i]] = i
        for addr in self.__edges.keys():
            for neigh in self.__edges[addr].content():
                if pos[addr] > pos[neigh]:
                    return True
        return False
    
    def topo(self):
        return self.__topo