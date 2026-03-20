import numpy as np
import random

# ------------------------------------
# Maze Environment
# ------------------------------------

maze = [
    ['S', 0, 0],
    [1, 1, 0],
    [0, 0, 'G']
]

rows = 3
cols = 3

start = (0,0)
goal = (2,2)

actions = ["up","down","left","right"]

# ------------------------------------
# Function to take action
# ------------------------------------

def step(state, action):

    r,c = state

    if action == "up":
        r -= 1
    elif action == "down":
        r += 1
    elif action == "left":
        c -= 1
    elif action == "right":
        c += 1

    # boundary check
    if r<0 or r>=rows or c<0 or c>=cols:
        return state, -1, False

    # wall check
    if maze[r][c] == 1:
        return state, -1, False

    new_state = (r,c)

    if new_state == goal:
        return new_state, 10, True

    return new_state, -0.1, False


# ------------------------------------
# Q Table
# ------------------------------------

Q = {}

for r in range(rows):
    for c in range(cols):
        Q[(r,c)] = {a:0 for a in actions}

# ------------------------------------
# Q-learning parameters
# ------------------------------------

alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 500

# ------------------------------------
# Training
# ------------------------------------

for ep in range(episodes):

    state = start
    done = False

    while not done:

        # exploration vs exploitation
        if random.uniform(0,1) < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[state], key=Q[state].get)

        next_state, reward, done = step(state,action)

        best_next = max(Q[next_state].values())

        Q[state][action] = Q[state][action] + alpha*(reward + gamma*best_next - Q[state][action])

        state = next_state

# ------------------------------------
# Testing the learned policy
# ------------------------------------

print("\nLearned Q-values:")
for state in Q:
    print(state, Q[state])

print("\nAgent Navigation:\n")

state = start
done = False
step_count = 0

while not done:

    action = max(Q[state], key=Q[state].get)
    next_state, reward, done = step(state, action)

    print(f"Step {step_count+1}")
    print("Current State:", state)
    print("Action Taken:", action)
    print("Next State:", next_state)
    print("Reward:", reward)
    print("----------------------")

    state = next_state
    step_count += 1

print("\nGoal Reached!")
print("Total Steps:", step_count)
