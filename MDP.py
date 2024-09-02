import numpy as np
import time


def initialize_values(w, h, L):
    values = np.zeros((h, w)) 
    for (x, y, reward) in L:
        values[h -1 - y, x] = reward 
    return values


def value_iteration(w, h, L, p, r, gamma=0.5, epsilon=0.01):
    s_time = time.time()

    values = initialize_values(w, h, L)
    delta = float('inf')
    walls = [(x, y) for (x, y, v) in L if v == 0]  
    while delta > epsilon:
        if time.time() - s_time > 600:
            return values
        new_values = np.copy(values)
        delta = 0
        for y in range(h): 
            for x in range(w): 
                if (x, h - 1 - y) in [(pos[0], pos[1]) for pos in L]:
                    continue  
                v = values[y, x]
                max_value = float('-inf')
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Directions: left, right, down, up
                    new_x, new_y = x + dx, y + dy
                    if (new_x, h - 1 - new_y) in walls or not (0 <= new_x < w and 0 <= new_y < h):
                        success_value = p * values[y, x]  # Collision with wall
                    else:
                        success_value = p * values[new_y, new_x]
                    
                    # Handle failure cases with checks for out-of-bounds or walls
                    if (dx, dy) == (-1, 0):  # Left
                        fail_values = []
                        if 0 <= y + 1 < h and not (y + 1, x) in walls :
                            fail_values.append(values[y + 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls :
                            fail_values.append(values[y - 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                    elif (dx, dy) == (1, 0):  # Right
                        fail_values = []
                        if 0 <= y + 1 < h and not (y + 1, x) in walls :
                            fail_values.append(values[y + 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls :
                            fail_values.append(values[y - 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                    elif (dx, dy) == (0, -1):  # Down
                        fail_values = []
                        if 0 <= x + 1 < w and not (y, h-1-(x+1)) in walls :
                            fail_values.append(values[y, x + 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= x - 1 < w and not (y, h-1-(x-1)) in walls :
                            fail_values.append(values[y, x - 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                    elif (dx, dy) == (0, 1):  # Up
                        fail_values = []
                        if 0 <= x + 1 < w and not (y, h-1-(x+1)) in walls :
                            fail_values.append(values[y, x + 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= x - 1 < w and not (y, h-1-(x-1)) in walls :
                            fail_values.append(values[y, x - 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                    
                    total_value = r + gamma*(success_value + fail_value)
                    max_value = max(max_value, total_value)
                new_values[y, x] =  max_value
                delta = max(delta, abs(v - new_values[y, x]))
                values = new_values
    return values



def extract_policy(values, w, h, L, p, r, gamma=0.5):
    policy = np.full((h, w), None)
    for (x, y, reward) in L:
        policy[h -1 - y, x] = reward 
    actions = ['L', 'R', 'U', 'D']
    walls = [(x, y) for (x, y, v) in L if v == 0]  # Define walls based on L
    for y in range(h):
        for x in range(w):
            if (x, h - 1 - y) in [(pos[0], pos[1]) for pos in L]:
                continue  # Skip cells with rewards/penalties
            best_action = None
            max_value = float('-inf')
            for action, (dx, dy) in zip(actions, [(-1, 0), (1, 0), (0, -1), (0, 1)]):  # Directions: left, right, down, up
                new_x, new_y = x + dx, y + dy
                if (new_x, h - 1 - new_y) in walls or not (0 <= new_x < w and 0 <= new_y < h):
                        success_value = p * values[y, x]  # Collision with wall
                else:
                        success_value = p * values[new_y, new_x]
                    
                 
                 # Handle failure cases with checks for out-of-bounds or walls
                if (dx, dy) == (-1, 0):  # Left
                        fail_values = []
                        if 0 <= y + 1 < h and not (y + 1, x) in walls :
                            fail_values.append(values[y + 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls :
                            fail_values.append(values[y - 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                elif (dx, dy) == (1, 0):  # Right
                        fail_values = []
                        if 0 <= y + 1 < h and not (y + 1, x) in walls :
                            fail_values.append(values[y + 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls :
                            fail_values.append(values[y - 1, x])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                elif (dx, dy) == (0, -1):  # Down
                        fail_values = []
                        if 0 <= x + 1 < w and not (y, h-1-(x+1)) in walls :
                            fail_values.append(values[y, x + 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= x - 1 < w and not (y, h-1-(x-1)) in walls :
                            fail_values.append(values[y, x - 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                elif (dx, dy) == (0, 1):  # Up
                        fail_values = []
                        if 0 <= x + 1 < w and not (y, h-1-(x+1)) in walls :
                            fail_values.append(values[y, x + 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        if 0 <= x - 1 < w and not (y, h-1-(x-1)) in walls :
                            fail_values.append(values[y, x - 1])
                        else:
                            fail_values.append(values[y, x])  # Collision with wall
                        fail_value = (1 - p) * 0.5 * sum(fail_values)
                
                total_value = r + gamma*(success_value + fail_value)

                if total_value > max_value:
                    max_value = total_value
                    best_action = action
            policy[y, x] = best_action
    return policy


def calculate_optimal_policy(w, h, L, p, r, gamma=0.5, epsilon=0.01):
    # Calculate optimal values
    optimal_values = value_iteration(w, h, L, p, r, gamma, epsilon)
    print("Optimal Values:")
    print(optimal_values)

    # Calculate optimal policy
    optimal_policy = extract_policy(optimal_values, w, h, L, p, r, gamma)
    print("Optimal Policy:")
    for row in optimal_policy:
        print(' '.join([str(elem) if elem is not None else '.' for elem in row]))
    return optimal_values,optimal_policy
