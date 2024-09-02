import numpy as np
import random
import time


def is_valid_state(state, L):
    for pos in L:
        if state[0] == pos[0] and state[1] == pos[1]:
            return False
    return True


def take_action(state, action, w, h, p, walls):
    direction_prob = random.uniform(0, 1)
    y, x = state
    if direction_prob < p:
        # פעולה ראשית בהסתברות p
        if action == 0 and x > 0 and (y, x - 1) not in walls:
            x -= 1
        elif action == 1 and x < w - 1 and (y, x + 1) not in walls:
            x += 1
        elif action == 3 and y < h - 1 and (y + 1, x) not in walls:
            y += 1
        elif action == 2 and y > 0 and (y - 1, x) not in walls:
            y -= 1
    else:
        # פעולה מאונכת בהסתברות (1-p)/2 לכל כיוון מאונך
        perpendicular_prob = random.uniform(0, 1)
        if action in [0,1]:
            if perpendicular_prob < 0.5 and y > 0 and (y - 1, x) not in walls:
                y -= 1  # למעלה
            elif y < h - 1 and (y + 1, x) not in walls:
                y += 1  # למטה
        elif action in [2,3]:
            if perpendicular_prob < 0.5 and x > 0 and (y, x - 1) not in walls:
                x -= 1  # שמאלה
            elif x < w - 1 and (y, x + 1) not in walls:
                x += 1  # ימינה

    return (y, x)


def epsilon_greedy_policy(Q, state, epsilon):
    action = ['L', 'R', 'U', 'D']
    if np.random.rand() < epsilon:
        return random.choice([0,1,2,3])  # Explore: random action
    else:
        return np.argmax(Q[state[0],state[1]])  # Exploit: action with max Q-value


def mainTraningQ(w, h, L, p, r, epsilon=0.01, gamma=0.5, alpha=0.1, num_episodes=1000):
    s_time = time.time()

    Q = np.zeros((h,w, 4))
    walls = [(y, x) for (y, x, v) in L if v == 0]


    for episode in range(num_episodes):

        state = (random.randint(0, h - 1), random.randint(0, w - 1))  # מצב התחלתי אקראי
        while not is_valid_state(state, L):
            state = (random.randint(0, h - 1), random.randint(0, w - 1))

        action = epsilon_greedy_policy(Q, state, epsilon)
        while is_valid_state(state, L):
            if time.time() - s_time > 600:
                return Q
            (y, x) = state

            next_state = take_action(state, action, w, h, p, walls)

            next_action = epsilon_greedy_policy(Q, next_state, epsilon)


            if next_state == (y, x):
                reward = -0.04
            else:
                reward = next((reward for (y, x, reward) in L if (y, x) == state), r)

            # SARSA update rule
            Q[y,x, action] = Q[y,x, action] + alpha * (
                        reward + gamma * Q[next_state[0],next_state[1], next_action] - Q[y,x, action])
            print(state, next_state)
            state = next_state
            action = next_action

    print("Training completed.")

    # Display the final Q-table
    print("Final Q-Table values")
    print(Q)
    return Q


def extract_policy2(arrayQ, w, h, L, transitions, rewards, gamma=0.5):
    optimal_values = np.zeros((h,w))
    policy = np.full((h, w), None)
    for (y, x, reward) in L:
        if 0 <= y < h and 0 <= x < w:
            policy[y, x] = reward
            optimal_values[y,x] = reward
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
                numberOfAction = np.argmax(arrayQ[y, x, :])  # Exploit: action with max Q-value
                optimal_values[y,x] = np.max(arrayQ[y,x,:])
            if numberOfAction == 0:
                best_action = 'L'
            if numberOfAction == 1:
                best_action = 'R'
            if numberOfAction == 2:
                best_action = 'U'
            if numberOfAction == 3:
                best_action = 'D'
            policy[y, x] = best_action
    return policy, optimal_values


def convert_L(L, h):
    return [(h - 1 - y, x, reward) for (x, y, reward) in L]


def calculate_optimal_policy(w, h, L, p, r, gamma=0.5, epsilon=0.01):
    L_new = convert_L(L, h)
    arrayQ = mainTraningQ(w, h, L_new, p, r, gamma)
    optimal_policy, optimal_values = extract_policy2(arrayQ, w, h, L_new, p, r, gamma)
    print("Optimal Values:")
    for row in optimal_values:
        print(' '.join([f"{elem:.4f}" if elem is not None else '.' for elem in row]))
    print("Optimal Policy:")
    for row in optimal_policy:
        print(' '.join([str(elem) if elem is not None else '.' for elem in row]))
    return optimal_values, optimal_policy


