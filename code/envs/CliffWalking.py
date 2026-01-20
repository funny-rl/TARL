from envs.gridcore import FallEnv

class CliffWalking(FallEnv):
    def __init__(self):
        pits = []
        for r in [7, 8, 9]:
            for c in range(1, 19):
                pits.append([r, c])
        shape: list[int] = [10, 20]
        start: list[int] = [9, 0]
        goal: list[int] = [9, 19]

        super(CliffWalking, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )