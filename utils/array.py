from typing import List, T


class NoDuplicateArray:
    def __init__(self):
        self._ = []

    def add(self, data):
        if self.exist(data):
            raise ValueError("Đã tồn tại! Không thể thêm")
        
        # Dùng kĩ thuật hai con trỏ để lập trình
        l = 0
        while l + 1 <= len(self._) and self._[l] <= data:
            l += 1

        self._.insert(l, data)

    def delete(self, data):
        m = self.search(data)
        if m == -1:
            raise ValueError("Đơn vị chưa tồn tại! Không thể xóa!")
        self._.pop(m)

    def search(self, data):
        l, r = 0, len(self._) - 1
        while l <= r:
            m = (l + r) // 2
            if self._[m] == data:
                return m
            if self._[m] < data:
                l = m + 1
            else:
                r = m - 1 
        return -1

    def exist(self, data):
        return not self.search(data) == -1

    def content(self) -> List[T]:
        return self._