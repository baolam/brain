import sys
sys.path.append("../")

from utils.array import NoDuplicateArray
array = NoDuplicateArray()

while True:
    data = input("Nhập dữ liệu nào: ")
    if data == "search":
        code = input("Nhập dữ liệu muốn tìm kiếm: ")
        print(array.search(code))
    elif data == "delete":
        code = input("Nhập dữ liệu muốn xóa: ")
        array.delete(code)
        print(array.content())
    else:
        array.add(data)
        print(array.content())