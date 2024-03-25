from typing import List
from ..graph.ForwardGraph import ForwardGraph, Graph
from .Learning import Learning


class LearnForwardUnit(Learning):
    def __init__(self, target: ForwardGraph, ac_learn : List[Graph.UNIT_TYPE] = []):
        '''
        Học tập được thực hiện trên một đơn vị trong một chuỗi lan truyền
        '''
        super().__init__(target)
        for unit in self._target._units.alls():
            unit.set_learn(False)

        for _unit in ac_learn:
            self._target._units.get(_unit) \
                .set_learn(True)