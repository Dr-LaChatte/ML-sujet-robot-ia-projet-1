"""
Module agents - Agents d'apprentissage (Q-Learning et k-NN).
"""

from src.agents.q_learning_agent import QLearningAgent
from src.agents.knn_agent import KNNAgent, find_best_k

__all__ = ['QLearningAgent', 'KNNAgent', 'find_best_k']
