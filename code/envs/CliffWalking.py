from envs.gridcore import FallEnv

class CliffWalking(FallEnv):
    def __init__(self):
        pits : list[list[int,int]] = [
            [3,1],
            [3,2],
            [3,3],
            [3,4],
            [3,5],
            [3,6],
            [3,7],
            [3,8],
            [3,9],
            [3,10],
        ]
        shape: list[int] = [4, 12]
        start: list[int] = [3, 0]
        goal: list[int] = [3, 11]

        super(CliffWalking, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
            max_steps=1000
        )
        pass