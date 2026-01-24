import collections

def get_optimal_repetitions(pits_list, shape, goal, MAX_REPETITION) -> dict:

    R, C = shape[0], shape[1]
    pits = set((r, c) for r, c in pits_list)

    actions = {
        1: (-1, 0), # Up
        2: (0, 1),  # Right
        3: (1, 0),  # Down
        4: (0, -1)  # Left
    }

    def is_valid(r, c):
        return 0 <= r < R and 0 <= c < C

    # Helper: State Index 변환
    def to_idx(r, c):
        return r * C + c

    dist_map = {} # (r,c) -> distance
    queue = collections.deque([(goal[0], goal[1], 0)])
    visited = set([(goal[0], goal[1])])
    dist_map[(goal[0], goal[1])] = 0
    
    while queue:
        r, c, d = queue.popleft()
        
        for move_r, move_c in actions.values():
            nr, nc = r + move_r, c + move_c
            if is_valid(nr, nc) and (nr, nc) not in visited and (nr, nc) not in pits:
                visited.add((nr, nc))
                dist_map[(nr, nc)] = d + 1
                queue.append((nr, nc, d + 1))

    optimal_rep_dict = {}

    for r in range(R):
        for c in range(C):
            state_idx = to_idx(r, c)

            if (r, c) in pits or (r, c) == goal:
                continue

            current_dist = dist_map.get((r, c), float('inf'))
            optimal_rep_dict[state_idx] = {}
            for action_idx, (dr, dc) in actions.items():
                nr, nc = r + dr, c + dc

                if is_valid(nr, nc) and (nr, nc) in pits:
                    continue

                next_dist = dist_map.get((nr, nc), float('inf'))
                
                if not is_valid(nr, nc) or next_dist >= current_dist:
                    optimal_rep_dict[state_idx][action_idx] = [1]
                    continue

                valid_repetitions = []
                min_one_direction = 0
                goal_flag = False

                sim_r, sim_c = r, c
                for n in range(1, MAX_REPETITION + 1):
                    prev_dist = dist_map.get((sim_r, sim_c), float('inf'))
                    sim_r, sim_c = sim_r + dr, sim_c + dc
                    curr_dist = dist_map.get((sim_r, sim_c), float('inf'))
                    if prev_dist < curr_dist:
                        break
                    if is_valid(sim_r, sim_c) and (sim_r, sim_c) == goal:
                        valid_repetitions.extend(range(n, MAX_REPETITION + 1))
                        goal_flag = True
                        break
                    if not is_valid(sim_r, sim_c) or (sim_r, sim_c) in pits:
                        break
                    optimal_count = 0
                    for _, (n_dr, n_dc) in actions.items():
                        check_r, check_c = sim_r + n_dr, sim_c + n_dc
                        if is_valid(check_r, check_c):
                            check_dist = dist_map.get((check_r, check_c), float('inf'))
                            if check_dist < dist_map.get((sim_r, sim_c), float('inf')):
                                optimal_count += 1
                                
                    if optimal_count == 2:
                        valid_repetitions.append(n)
                    elif optimal_count == 1:
                        min_one_direction = n
                if min_one_direction > 0 and not goal_flag:
                    valid_repetitions.append(min_one_direction)
                optimal_rep_dict[state_idx][action_idx] = sorted(list(valid_repetitions))
    return optimal_rep_dict

if __name__ == "__main__":
    result = get_optimal_repetitions()
    for state_idx in sorted(result.keys()):
        print(f"State {state_idx}: {result[state_idx]}")

    














