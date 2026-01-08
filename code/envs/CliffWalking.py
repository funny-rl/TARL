from envs.gridcore import FallEnv

class CliffWalking(FallEnv):
    def __init__(self):
        pits : list[list[int,int]] = [
            [5,1],
            [5,2],
            [5,3],
            [5,4],
            [5,5],
            [5,6],
            [5,7],
            [5,8],
            
            [4,1],
            [4,2],
            [4,3],
            [4,4],
            [4,5],
            [4,6],
            [4,7],
            [4,8],
        ]
        shape: list[int] = [6, 10]
        start: list[int] = [5, 0]
        goal: list[int] = [5, 9]

        super(CliffWalking, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )