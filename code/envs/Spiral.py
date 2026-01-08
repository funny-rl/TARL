from envs.gridcore import FallEnv

class Spiral(FallEnv):
    def __init__(self):

        shape : list[int] = [11, 11]
        path_coords: list[list[int, int]] = []

        shape_x = shape[0]
        shape_y = shape[1]
        
        start : list[int] = [shape_x // 2, shape_y // 2]
        curr_r, curr_c = start[0], start[1]

        directions = [
            (0, 1),  # right
            (-1, 0), # up
            (0, -1), # left
            (1, 0)   # down
        ]
        step_size = 1
        dir_index = 0


        while 0 <= curr_r < shape_x and 0 <= curr_c < shape_y:
            dr, dc = directions[dir_index]
            for _ in range(step_size):
                if 0 <= curr_r < shape_x and 0 <= curr_c < shape_y:
                    path_coords.append([curr_r, curr_c])
                    curr_r += dr
                    curr_c += dc
                else:
                    break
            dir_index = (dir_index + 1) % 4
            if not (0 <= curr_r < shape_x and 0 <= curr_c < shape_y):
                break
            step_size += 1
        
        goal : list[int] = path_coords[-1]
        pits : list[list[int,int]] = [
            [r, c] for r in range(shape_x) for c in range(shape_y)
            if [r, c] not in path_coords
        ]

        super(Spiral, self).__init__(
            pits=pits,
            shape=shape,
            start=start,
            goal=goal,
            max_steps=1000,
            dense_reward=True,
        )