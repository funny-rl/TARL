from envs.gridcore import FallEnv

class Bridge(FallEnv):
    def __init__(self):
        pits = []

        for r in [0, 1]:
            for c in range(3, 17):
                pits.append([r, c])

        # 하단 pit 영역 (rows 8~9)
        for r in [8, 9]:
            for c in range(3, 17):
                pits.append([r, c])

        shape = [10, 20]    # (height=10, width=20)
        start = [1, 0]
        goal = [1, 19]

        super(Bridge, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
        )