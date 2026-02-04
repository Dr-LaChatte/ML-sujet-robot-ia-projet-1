import argparse
import sys
from src.q_learning import train_agent, QLearningAgent
from src.generate_dataset import generate_dataset
from src.supervised_learning import train_supervised, SupervisedAgent
from src.compare import compare_agents
from src.environment import WarehouseEnv

def play(agent_type, length=50):
    env = WarehouseEnv(render_mode=True, length=length)

    if agent_type == 'rl':
        agent = QLearningAgent()
        try:
            agent.load("data/q_table.npy")
            agent.epsilon = 0.0
            print("Loaded RL Agent.")
        except:
            print("Could not load RL Agent. Using random.")
    elif agent_type == 'sl':
        agent = SupervisedAgent("data/knn_model.pkl")
        print("Loaded SL Agent.")
    else:
        print("Unknown agent type")
        return

    while True:
        state = env.reset()
        done = False
        print("New Episode")
        while not done:
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            env.render()

        print(f"Episode Finished. Reward: {reward}")

def main():
    parser = argparse.ArgumentParser(description="Navette Robotique - IA Project")
    parser.add_argument('mode', choices=['train_rl', 'gen_data', 'train_sl', 'compare', 'play_rl', 'play_sl'], help="Mode to run")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of episodes for training/comparison")

    args = parser.parse_args()

    if args.mode == 'train_rl':
        train_agent(episodes=args.episodes)
    elif args.mode == 'gen_data':
        generate_dataset(n_samples=args.episodes) # Reusing episodes arg for samples roughly
    elif args.mode == 'train_sl':
        train_supervised()
    elif args.mode == 'compare':
        compare_agents(episodes=args.episodes)
    elif args.mode == 'play_rl':
        play('rl')
    elif args.mode == 'play_sl':
        play('sl')

if __name__ == "__main__":
    main()
