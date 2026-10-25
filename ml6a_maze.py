import numpy as np
import random

maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 2]
])

start = (0, 0)
goal = (4, 4)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
q_table = np.zeros((5, 5, 4))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 3000

def move(state, action):
    x, y = state
    dx, dy = actions[action]
    nx, ny = x + dx, y + dy
    if (nx < 0 or nx >= 5 or ny < 0 or ny >= 5
        or maze[nx, ny] == 1):
        return state, -10
    if (nx, ny) == goal:
        return (nx, ny), 100
    return (nx, ny), -1

# Training
for episode in range(episodes):
    state = start
    for step in range(100):
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state])
        next_state, reward = move(state, action)
        old_q = q_table[state][action]
        next_max = np.max(q_table[next_state])
        q_table[state][action] = old_q + alpha * (
            reward + gamma * next_max - old_q
        )
        state = next_state
        if state == goal:
            break

# Testing learned path
state = start
path = [state]
for step in range(30):
    action = np.argmax(q_table[state])
    next_state, reward = move(state, action)
    if next_state == state or next_state in path:
        break
    state = next_state
    path.append(state)
    if state == goal:
        break

print("Learned Path:")
print(path)

print("\nMaze:")
for i in range(5):
    for j in range(5):
        if (i, j) == start:
            print("S", end=" ")
        elif (i, j) == goal:
            print("G", end=" ")
        elif (i, j) in path:
            print("*", end=" ")
        elif maze[i, j] == 1:
            print("#", end=" ")
        else:
            print(".", end=" ")
    print()