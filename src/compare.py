import numpy as np
from src.environment import WarehouseEnv
from src.q_learning import QLearningAgent
from src.supervised_learning import SupervisedAgent

def evaluate_agent(agent, episodes=100, length=50):
    env = WarehouseEnv(length=length)
    collisions = 0
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Polymorphism: both agents need choose_action(state)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward

            if reward == -100: # Collision reward
                collisions += 1
            if done and reward > 0: # Success (Goal reached)
                successes += 1

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    success_rate = successes / episodes
    collision_rate = collisions / episodes # Note: this might be > 100% if we counted multiple collisions per episode, but env sets done=True on collision.

    return avg_reward, success_rate, collision_rate

def compare_agents(episodes=100):
    print(f"Comparing Agents over {episodes} episodes...")

    # Load RL Agent
    rl_agent = QLearningAgent()
    rl_agent.load("data/q_table.npy")
    rl_agent.epsilon = 0.0

    # Load SL Agent
    sl_agent = SupervisedAgent("data/knn_model.pkl")

    # Evaluate RL
    rl_avg, rl_succ, rl_coll = evaluate_agent(rl_agent, episodes)
    print(f"[RL Agent] Avg Reward: {rl_avg:.2f}, Success Rate: {rl_succ:.2f}, Collision Rate: {rl_coll:.2f}")

    # Evaluate SL
    sl_avg, sl_succ, sl_coll = evaluate_agent(sl_agent, episodes)
    print(f"[SL Agent] Avg Reward: {sl_avg:.2f}, Success Rate: {sl_succ:.2f}, Collision Rate: {sl_coll:.2f}")

if __name__ == "__main__":
    compare_agents()
