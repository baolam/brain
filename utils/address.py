SPLIT_CHAR = '$'

from uuid import uuid4
counter = 0

def set_counter(c):
    global counter
    counter = c

def get_counter():
    global counter
    return counter

def get():
    global counter
    counter += 1
    return str(counter) + SPLIT_CHAR + str(uuid4())