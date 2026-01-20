from envs.gridcore import FallEnv

class Bridge(FallEnv):
    def __init__(self):
        pits = []

        for r in [0, 1, 2]:
            for c in range(2, 7):
                pits.append([r, c])

        # 하단 pit 영역 (rows 8~9)
        for r in [5, 6, 7]:
            for c in range(2, 7):
                pits.append([r, c])

        shape = [8, 10]    # (height=8, width=10)
        start = [0, 0]
        goal = [0, 9]

        super(Bridge, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )