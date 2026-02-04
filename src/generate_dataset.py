import numpy as np
import pandas as pd
import os
from src.environment import WarehouseEnv
from src.q_learning import QLearningAgent

def generate_dataset(q_table_path="data/q_table.npy", output_path="data/dataset.csv", n_samples=10000):
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table not found at {q_table_path}")
        return

    agent = QLearningAgent()
    agent.load(q_table_path)
    agent.epsilon = 0.0 # Exploit only

    env = WarehouseEnv(length=50)

    data = []
    samples_collected = 0

    episode = 0
    while samples_collected < n_samples:
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)

            # Record state and action
            # state is (shuttle_y, dist_state, obs_y)
            data.append({
                "shuttle_y": state[0],
                "dist_state": state[1],
                "obs_y": state[2],
                "action": action
            })
            samples_collected += 1

            next_state, reward, done = env.step(action)
            state = next_state

            if samples_collected >= n_samples:
                break

        episode += 1
        if episode % 100 == 0:
            print(f"Collected {samples_collected} samples...")

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} with {len(df)} samples.")

if __name__ == "__main__":
    generate_dataset()
