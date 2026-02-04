from src.q_learning import train_agent
import os

def test_training():
    print("Testing training...")
    train_agent(episodes=200000, save_path="data/test_q_table.npy")

    if os.path.exists("data/test_q_table.npy") and os.path.exists("data/convergence_plot.png"):
        print("Training test passed: Files created.")
    else:
        print("Training test failed: Files missing.")

if __name__ == "__main__":
    test_training()
