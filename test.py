
import gym
import numpy as np

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000
max_steps_per_episode = 500

# Discretize the state space
num_bins = 40
state_bins = [np.linspace(-2.4, 2.4, num_bins),
              np.linspace(-3.0, 3.0, num_bins),
              np.linspace(-0.5, 0.5, num_bins),
              np.linspace(-2.0, 2.0, num_bins)]

# Initialize the Q-table with random values
num_actions = env.action_space.n
q_table = np.random.uniform(low=-1, high=1, size=([num_bins] * len(env.observation_space.high) + [num_actions]))

# Epsilon-greedy exploration
epsilon = 0.2

# Q-learning training
for episode in range(num_episodes):
    state = env.reset()
    state_bins_indices = [np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))]
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state_bins_indices])  # Exploit

        new_state, reward, done, _ = env.step(action)
        new_state_bins_indices = [np.digitize(new_state[i], state_bins[i]) - 1 for i in range(len(new_state))]

        # Update Q-value using Q-learning formula
        q_table[state_bins_indices][action] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state_bins_indices]) - q_table[state_bins_indices][action])

        state = new_state
        state_bins_indices = new_state_bins_indices
        total_reward += reward

        if done:
            break

    # Print the total reward for this episode
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Test the learned policy
total_rewards = 0
num_test_episodes = 100

for _ in range(num_test_episodes):
    state = env.reset()
    state_bins_indices = [np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))]
    done = False

    while not done:
        action = np.argmax(q_table[state_bins_indices])
        state, _, done, _ = env.step(action)
        state_bins_indices = [np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))]

    total_rewards += 1 if done else 0

average_reward = total_rewards / num_test_episodes
print(f"Average Test Reward: {average_reward}")
