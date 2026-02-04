import numpy as np
import random
import matplotlib.pyplot as plt
import os
from src.environment import WarehouseEnv

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # State space: Shuttle Y (3) x Dist (3) x Obs Y (3)
        self.q_table = np.zeros((3, 3, 3, 3))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)

def train_agent(episodes=1000, length=50, save_path="data/q_table.npy"):
    env = WarehouseEnv(length=length)
    agent = QLearningAgent()

    rewards_history = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (e + 1) % 100 == 0:
            print(f"Episode {e+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")

    # Save Q-table
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Q-table saved to {save_path}")

    # Plot
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Convergence")
    plt.savefig("data/convergence_plot.png")
    print("Convergence plot saved to data/convergence_plot.png")

    return agent

if __name__ == "__main__":
    train_agent(episodes=5000)
