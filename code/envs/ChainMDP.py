from envs.gridcore import FallEnv

class ChainMDP(FallEnv):
    def __init__(self):
        pits : list[list[int,int]] = []
        shape: list[int] = [1, 100]
        start: list[int] = [0, 19]
        goal: list[int] = [0, 99]

        super(ChainMDP, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
            max_steps=2000,
        )