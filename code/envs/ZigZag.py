from envs.gridcore import FallEnv

class ZigZag(FallEnv):
    def __init__(self):
        pits : list[list[int,int]] = [
            [0,2],
            [1,2],
            [2,2],
            [3,2],
            [4,2],
            [5,2],
            
            [0,3],
            [1,3],
            [2,3],
            [3,3],
            [4,3],
            [5,3],
            
            [3,6],
            [4,6],
            [5,6],
            [6,6],
            [7,6],
            
            [3,7],
            [4,7],
            [5,7],
            [6,7],
            [7,7],
        ]
        shape: list[int] = [8, 10]
        start: list[int] = [0, 0]
        goal: list[int] = [7, 9]

        super(ZigZag, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )