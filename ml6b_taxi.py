import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create Taxi environment
env = gym.make("Taxi-v4")

# Initialize Q-table
q_table = np.zeros((
    env.observation_space.n,
    env.action_space.n
))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
episodes = 10000
rewards = []

# Training
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        target = reward
        if not done:
            target += gamma * np.max(q_table[next_state])
            
        q_table[state, action] += alpha * (
            target - q_table[state, action]
        )
        
        state = next_state
        total_reward += reward
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Training Completed!")

# Test trained agent
state, info = env.reset(seed=42)
done = False
total_reward = 0
steps = 0

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1

print("Total Reward:", total_reward)
print("Steps Taken:", steps)

# Plot training rewards
window = 100
moving_avg = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.plot(moving_avg)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Taxi Q-Learning Performance")
plt.grid(True)
plt.show()
env.close()