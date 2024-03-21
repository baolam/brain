from learn.LearnGraph import LearnGraph
from .Learning import Learning


class LearnUnit(Learning):
    def __init__(self, target: LearnGraph):
        super().__init__(target)