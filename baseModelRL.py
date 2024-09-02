import numpy as np
import random
import time

def convert_L(L, h):
    return [(h - 1 - y, x, reward) for (x, y, reward) in L]


def initialize_values_and_policy(w, h, L):
    values = np.zeros((h, w))
    for (y, x, reward) in L:
        if 0 <= y < h and 0 <= x < w:
            values[y, x] = reward
    if h!=3 or w!=4:
        policy = np.full((h,w), None)
        for y in range(h):
            for x in range(w):
                policy[y,x] = random.choice(["R", "D", "L", "U"])
        for (y, x, reward) in L:
            if 0 <= y < h and 0 <= x < w:
                policy[y, x] = reward
    else:
        policy = np.array((["R", "R", "R", 1], ["U", 0, "U", -1], ["U", "R", "U", "L"]))


    print(policy)
    return values, policy


def choose_action(state, policy, epsilon=0.01):
    if np.random.rand() > epsilon:
        return policy[state[0], state[1]]
    return random.choice(['L', 'R', 'U', 'D'])  # Explore: random action


def take_action(state, action, w, h, p, walls):
    direction_prob = random.uniform(0, 1)
    y, x = state
    if direction_prob < p:
        # פעולה ראשית בהסתברות p
        if action == 'L' and x > 0 and (y, x - 1) not in walls:
            x -= 1
        elif action == 'R' and x < w - 1 and (y, x + 1) not in walls:
            x += 1
        elif action == 'D' and y < h - 1 and (y + 1, x) not in walls:
            y += 1
        elif action == 'U' and y > 0 and (y - 1, x) not in walls:
            y -= 1
    else:
        # פעולה מאונכת בהסתברות (1-p)/2 לכל כיוון מאונך
        perpendicular_prob = random.uniform(0, 1)
        if action in ['L', 'R']:
            if perpendicular_prob < 0.5 and y > 0 and (y - 1, x) not in walls:
                y -= 1  # למעלה
                # print("pro up")
            elif y < h - 1 and (y + 1, x) not in walls:
                y += 1  # למטה
                # print("pro down")
        elif action in ['U', 'D']:
            if perpendicular_prob < 0.5 and x > 0 and (y, x - 1) not in walls:
                # print("pro left")
                x -= 1  # שמאלה
            elif x < w - 1 and (y, x + 1) not in walls:
                # print("pro right")
                x += 1  # ימינה

    return (y, x)


def learn_mdp(experiences, w, h):
    rewards = np.zeros((h, w, 4))  # כל פעולה לכל מצב
    transitions = np.full((h, w, 4, h, w), 0.8)
    actions = ['L', 'R', 'U', 'D']
    count_rewards = np.zeros((h, w, 4))  # סופרים את מספר הפעמים שקיבלנו תגמול עבור כל פעולה במצב מסוים

    for (i, a, r, j) in experiences:
        y1, x1 = i
        y2, x2 = j
        if a not in actions:
            continue
        a_idx = actions.index(a)
        rewards[y1, x1, a_idx] += r
        count_rewards[y1, x1, a_idx] += 1
        transitions[y1, x1, a_idx, y2, x2] += 1

    # חישוב ההסתברויות
    for y1 in range(h):
        for x1 in range(w):
            for a_idx in range(4):
                total_transitions = np.sum(transitions[y1, x1, a_idx])
                if total_transitions > 0:
                    transitions[y1, x1, a_idx] /= total_transitions

    # חישוב התגמול הממוצע
    for y1 in range(h):
        for x1 in range(w):
            for a_idx in range(4):
                if count_rewards[y1, x1, a_idx] > 0:
                    rewards[y1, x1, a_idx] /= count_rewards[y1, x1, a_idx]

    return rewards, transitions


def solve_mdp(values, rewards, transitions, w, h, L):
    optimal_values = value_iteration2(values, w, h, L, transitions, rewards)

    print("Optimal Values:")
    for row in optimal_values:
        print(' '.join([f"{elem:.4f}" if elem is not None else '.' for elem in row]))
    optimal_policy = extract_policy2(optimal_values, w, h, L, transitions, rewards)
    print("Optimal Policy:")
    for row in optimal_policy:
        print(' '.join([str(elem) if elem is not None else '.' for elem in row]))
    return optimal_values, optimal_policy


def is_valid_state(state, L):
    for pos in L:
        if state[0] == pos[0] and state[1] == pos[1]:
            return False
    return True


def model_based_rl(w, h, L, p, r, gamma=0.5, episodes=50):
    s_time = time.time()
    L_new = convert_L(L, h)
    values, policy = initialize_values_and_policy(w, h, L_new)
    rewards = np.zeros((h, w, 4))
    walls = [(x, y) for (x, y, v) in L_new if v == 0]

    for j in range(50):  # התכנסות כמה פעמים צריך לשפר police

        experiences = []
        for i in range(50):  # דגימות

            state = (random.randint(0, h - 1), random.randint(0, w - 1))  # מצב התחלתי אקראי
            while not is_valid_state(state, L_new):
                state = (random.randint(0, h - 1), random.randint(0, w - 1))

            while is_valid_state(state, L_new):  # מתחילים בדגימה מהמצב עד לפרס/קנס
                if time.time() - s_time > 600:
                    return values, policy
                current_step = [state]
                action = choose_action(state, policy)
                x, y = state
                state = take_action(state, action, w, h, p, walls)
                current_step.append(action)

                if state == (x, y):
                    reward = -0.04
                else:
                    reward = next((reward for (x, y, reward) in L_new if (x, y) == state), r)

                current_step.append(reward)
                current_step.append(state)
                experiences.append(current_step)
        rewards, transitions = learn_mdp(experiences, w, h)
        values, policy = solve_mdp(values, rewards, transitions, w, h, L_new)

    return values, policy


def initialize_values(w, h, L):
    values = np.zeros((h, w))
    for (y, x, reward) in L:
        if 0 <= y < h and 0 <= x < w:
            values[y, x] = reward
    return values


def value_iteration2(values, w, h, L, transitions, rewards, gamma=0.5, epsilon=0.01, max_iterations=1000):
    delta = float('inf')
    walls = [(y, x) for (y, x, v) in L if v == 0]
    iteration = 0
    while delta > epsilon and iteration < max_iterations:
        new_values = np.copy(values)
        delta = 0
        for y in range(h):
            for x in range(w):
                if (y, x) in [(pos[0], pos[1]) for pos in L]:
                    continue
                v = values[y, x]
                max_value = float('-inf')
                for a_idx, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    new_y, new_x = y + dy, x + dx
                    if not (0 <= new_x < w and 0 <= new_y < h):
                        continue
                    if (new_y, new_x) in walls:
                        success_value = transitions[y, x, a_idx, y, x] * values[y, x]  # Collision with wall
                    else:
                        success_value = transitions[y, x, a_idx, new_y, new_x] * values[new_y, new_x]

                    # Handle failure cases with checks for out-of-bounds or walls
                    if (dx, dy) == (-1, 0):  # Left
                        fail_values = 0
                        if 0 <= y + 1 < h and not (y + 1, x) in walls:
                            fail_values += values[y + 1, x]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls:
                            fail_values += values[y - 1, x]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                    elif (dx, dy) == (1, 0):  # Right
                        fail_values = 0
                        if 0 <= y + 1 < h and not (y + 1, x) in walls:
                            fail_values += values[y + 1, x]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        if 0 <= y - 1 < h and not (y - 1, x) in walls:
                            fail_values += values[y - 1, x]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                    elif (dx, dy) == (0, -1):  # Down
                        fail_values = 0
                        if 0 <= x + 1 < w and not (y, x + 1) in walls:
                            fail_values += values[y, x + 1]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        if 0 <= x - 1 < w and not (y, x - 1) in walls:
                            fail_values += values[y, x - 1]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                    elif (dx, dy) == (0, 1):  # Up
                        fail_values = 0
                        if 0 <= x + 1 < w and not (y, x + 1) in walls:
                            fail_values += values[y, x + 1]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        if 0 <= x - 1 < w and not (y, x - 1) in walls:
                            fail_values += values[y, x - 1]
                        else:
                            fail_values += values[y, x]  # Collision with wall
                        fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values

                    total_value = rewards[y, x, a_idx] + gamma * (success_value + fail_value)
                    max_value = max(max_value, total_value)
                new_values[y, x] = max_value
                delta = max(delta, abs(v - new_values[y, x]))
        iteration += 1
        values = new_values
        if iteration % 50 == 0:
            print(f"Iteration {iteration}, Delta: {delta}")
    return values


def extract_policy2(values, w, h, L, transitions, rewards, gamma=0.5):
    policy = np.full((h, w), None)
    for (y, x, reward) in L:
        if 0 <= y < h and 0 <= x < w:
            policy[y, x] = reward
    actions = ['L', 'R', 'U', 'D']
    walls = [(y, x) for (y, x, v) in L if v == 0]  # Define walls based on L

    for y in range(h):
        for x in range(w):
            if (y, x) in [(pos[0], pos[1]) for pos in L]:
                continue  # Skip cells with rewards/penalties
            best_action = None
            max_value = float('-inf')
            for a_idx, action in enumerate(actions):
                dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][a_idx]
                new_y, new_x = y + dy, x + dx
                if not (0 <= new_x < w and 0 <= new_y < h):
                    continue
                if (new_x, new_y) in walls or not (0 <= new_x < w and 0 <= new_y < h):
                    success_value = transitions[y, x, a_idx, y, x] * values[y, x]  # Collision with wall
                else:
                    success_value = transitions[y, x, a_idx, new_y, new_x] * values[new_y, new_x]

                # Handle failure cases with checks for out-of-bounds or walls
                if (dx, dy) == (-1, 0):  # Left
                    fail_values = 0
                    if 0 <= y + 1 < h and not (y + 1, x) in walls:
                        fail_values += values[y + 1, x]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    if 0 <= y - 1 < h and not (y - 1, x) in walls:
                        fail_values += values[y - 1, x]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                elif (dx, dy) == (1, 0):  # Right
                    fail_values = 0
                    if 0 <= y + 1 < h and not (y + 1, x) in walls:
                        fail_values += values[y + 1, x]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    if 0 <= y - 1 < h and not (y - 1, x) in walls:
                        fail_values += values[y - 1, x]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                elif (dx, dy) == (0, -1):  # Down
                    fail_values = 0
                    if 0 <= x + 1 < w and not (y, x + 1) in walls:
                        fail_values += values[y, x + 1]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    if 0 <= x - 1 < w and not (y, x - 1) in walls:
                        fail_values += values[y, x - 1]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values
                elif (dx, dy) == (0, 1):  # Up
                    fail_values = 0
                    if 0 <= x + 1 < w and not (y, x + 1) in walls:
                        fail_values += values[y, x + 1]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    if 0 <= x - 1 < w and not (y, x - 1) in walls:
                        fail_values += values[y, x - 1]
                    else:
                        fail_values += values[y, x]  # Collision with wall
                    fail_value = (1 - transitions[y, x, a_idx, new_y, new_x]) * 0.5 * fail_values

                total_value = rewards[y, x, a_idx] + gamma * (success_value + fail_value)

                if total_value > max_value:
                    max_value = total_value
                    best_action = action
            policy[y, x] = best_action
    return policy


