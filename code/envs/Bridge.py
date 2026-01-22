
from envs.gridcore import FallEnv

class Bridge(FallEnv):
    def __init__(self):
        pits : list[list[int,int]] = [
            [0,2],
            [1,2],
            [0,3],
            [1,3],
            [0,4],
            [1,4],
            [0,5],
            [1,5],
            [0,6],
            [1,6],
            [0,7],
            
            [1,7], # upper block
            [4,2],
            [5,2],
            [4,3],
            [5,3],
            [4,4],
            [5,4],
            [4,5],
            [5,5],
            [4,6],
            [5,6],
            [4,7],
            [5,7],

        ]
        shape: list[int] = [6, 10]
        start: list[int] = [0, 0]
        goal: list[int] = [0, 9]

        super(Bridge, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )
