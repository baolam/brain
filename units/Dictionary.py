from typing import Dict, Tuple, List
from torch import nn

from .Unit import Unit
from utils.address import SPLIT_CHAR
from utils.array import NoDuplicateArray


class Dictionary(nn.Module):
    def __init__(self, units : Dict[str, Unit]):
        '''
        Khởi tạo của Dictionary. Đóng vai trò quản lí các unit.
        Là một tập hợp bao gồm các Unit.
        '''
        super().__init__()
        self.__lock = False
        if units is None:
            self.__lock = True

        self.__units : Dict[str, Unit] = nn.ModuleDict(units)
    
    def exist(self, _unit : Tuple[Unit, str]):
        addr = _unit
        if isinstance(_unit, Unit):
            addr = _unit.name()

        try:
            self.__units[addr]
            return True
        except:
            return False

    def add(self, unit : Unit):
        assert isinstance(unit, Unit), "Không thể thêm đơn vị không kế thừa từ lớp Unit"
        if self.__lock:
            raise ValueError("Chức năng thêm đã bị khóa!")
        if self.exist(unit):
            raise ValueError("{} đã tồn tại. Không thể thêm!".format(unit.show_infor()))
        self.__units[unit.name()] = unit 

    def delete(self, unit : Tuple[Unit, str]):
        addr = unit
        if isinstance(unit, Unit):
            addr = unit.name()
        if not self.exist(addr):
            raise ValueError("Đơn vị với địa chỉ {} không được quản lí!".format(addr))
        if self.__lock:
            raise ValueError("Chức năng xóa đã bị khóa!")
        self.__units.pop(addr)

    def __bsearch_counter(self, counter : int, address : List[str]):
        l, r = 0, len(address) - 1
        counter = str(counter)

        while l <= r:
            m = (l + r) // 2
            _counter = address[m].split(SPLIT_CHAR)[0]
            if _counter == counter:
                return m
            if _counter < counter:
                l = m + 1
            else:
                r = m - 1
        
        return -1

    def __from_counter(self, counter : int):
        address = self.address()
        index = self.__bsearch_counter(counter, address)
        if index == -1:
            raise ValueError("Thứ tự {} không được quản lí!".format(counter))
        return self.__units[address[index]]

    def __from_address(self, addr : str):
        if not self.exist(addr):
            raise ValueError("{} không trong quản lí!".format(addr))
        return self.__units[addr]

    def get(self, addr : Tuple[str, int]) -> Unit:
        '''
        Trả về đơn vị dựa vào address hoặc từ counter 
        
        '''
        if isinstance(addr, str):
            return self.__from_address(addr)
        return self.__from_counter(addr)
    
    def address(self) -> List[str]:
        '''
        Trả về toàn bộ địa chỉ của các unit được sắp xếp theo thứ tự tăng dần
        '''
        return list(self.__units.keys()).sort()

    def layers(self) -> Dict[str, NoDuplicateArray]:
        '''
        Hàm trả về các lớp của từ điển
        '''
        layers = {  }
        for addr, unit in self.__units.items():
            layer = unit.layer()
            if layers.get(layer) is None:
                layers[layer] = NoDuplicateArray()
            layers[layer].add(addr)
        return layers
    
    def alls(self, return_addr : bool = False):
        '''
        Trả về toàn bộ đơn vị trong từ điển
        '''
        if return_addr:
            return list(self.__units.keys())
        return list(self.__units.values())