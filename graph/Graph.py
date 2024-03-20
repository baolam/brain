from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

from units.Dictionary import Dictionary
from utils.array import NoDuplicateArray
from units.Unit import Unit


class Graph(ABC):
    UNIT_TYPE = Tuple[str, Tuple[int, Unit]]

    def __init__(self, dictionary : Dictionary = None):
        if dictionary is None:
            dictionary = Dictionary()
        self._units = dictionary
        self.__initalize()

        self._edges : Dict[str, NoDuplicateArray] = { }

    def __initalize(self):
        for name in self._units.address():
            self._edges[name] = NoDuplicateArray()

    def add_unit(self, unit : Unit):
        self._units.add(unit)
        if self._edges.get(unit.name()) is None:
            self._edges[unit.name()] = NoDuplicateArray()

    def delete_unit(self, _unit : UNIT_TYPE):
        _unit = self.__address(_unit)
        # Xóa đơn vị ra khỏi từ điển
        self._units.delete(_unit)
        # Xóa đơn vị ra khỏi quản lí
        for neighbor in self.neighbor(_unit):
            self._edges[neighbor].delete(_unit)

    def __address(self, _unit : UNIT_TYPE) -> str:
        if isinstance(_unit, int):
            _unit = self._units.get(_unit).name()
        return _unit

    def neighbor(self, _unit : UNIT_TYPE):
        _unit = self.__address(_unit)
        neighbor = self._edges[_unit].content()
        return neighbor
    
    def add_edge(self, _from : UNIT_TYPE, _to : UNIT_TYPE):
        _from = self.__address(_from)
        _to = self.__address(_to)
        self._edges[_from].add(_to)

    def delete_edge(self, _from : UNIT_TYPE, _to : UNIT_TYPE):
        _from = self.__address(_from)
        _to = self.__address(_to)
        self._edges[_from].delete(_to)

    def edges(self) -> Dict[str, List[str]]:
        '''
        Trả về tập danh sách cạnh của đồ thi
        '''
        data = {  }
        for unit, neighbor in self._edges.items():
            data[unit] = neighbor.content()
        return data

    @abstractmethod
    def forward(self):
        pass