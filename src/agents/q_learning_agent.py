"""
Agent Q-Learning tabulaire pour la navette robotique.
Implémente l'algorithme Q-learning avec exploration epsilon-greedy.
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Dict
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    NUM_ACTIONS, NUM_LANES, NUM_DISTANCE_STATES,
    ACTION_NAMES, Q_TABLE_FILE
)


class QLearningAgent:
    """
    Agent Q-Learning tabulaire.
    
    Apprend une politique d'évitement d'obstacles par interaction
    avec l'environnement de simulation.
    """
    
    def __init__(
        self,
        num_states: int = None,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY
    ):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            num_states: Nombre d'états possibles
            num_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (α)
            discount_factor: Facteur de discount (γ)
            epsilon_start: Valeur initiale de epsilon
            epsilon_end: Valeur minimale de epsilon
            epsilon_decay: Facteur de décroissance de epsilon
        """
        # Calcul automatique du nombre d'états si non fourni
        if num_states is None:
            num_states = NUM_LANES * NUM_DISTANCE_STATES * NUM_LANES
        
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Hyperparamètres
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Table Q : initialisée à zéro
        self.q_table = np.zeros((num_states, num_actions))
        
        # Historique pour le suivi
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'collisions': [],
            'avoidances': []
        }
        
        # Compteurs
        self.total_steps = 0
        self.total_episodes = 0
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Choisit une action selon la politique epsilon-greedy.
        
        Args:
            state: État actuel (entier)
            training: Si True, utilise exploration epsilon-greedy
            
        Returns:
            Action choisie (0, 1, ou 2)
        """
        if training and random.random() < self.epsilon:
            # Exploration : action aléatoire
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation : meilleure action selon Q-table
            return int(np.argmax(self.q_table[state]))
    
    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Met à jour la table Q selon l'équation de Bellman.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: Nouvel état
            done: True si l'épisode est terminé
        """
        # Valeur Q actuelle
        current_q = self.q_table[state, action]
        
        # Valeur Q future
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Mise à jour de la table Q
        self.q_table[state, action] += self.learning_rate * (target_q - current_q)
        
        self.total_steps += 1
    
    def decay_epsilon(self):
        """Décroît epsilon après chaque épisode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def record_episode(
        self,
        total_reward: float,
        episode_length: int,
        collisions: int,
        avoidances: int
    ):
        """
        Enregistre les statistiques d'un épisode.
        
        Args:
            total_reward: Récompense totale de l'épisode
            episode_length: Durée de l'épisode (en pas)
            collisions: Nombre de collisions
            avoidances: Nombre d'évitements
        """
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(episode_length)
        self.training_history['epsilon_values'].append(self.epsilon)
        self.training_history['collisions'].append(collisions)
        self.training_history['avoidances'].append(avoidances)
        self.total_episodes += 1
    
    def get_policy(self) -> np.ndarray:
        """
        Retourne la politique apprise (action optimale pour chaque état).
        
        Returns:
            Array des meilleures actions pour chaque état
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_state_info(self, state: int) -> Dict:
        """
        Décode un état en ses composantes.
        
        Args:
            state: État encodé
            
        Returns:
            Dictionnaire avec les composantes de l'état
        """
        obstacle_lane = state % NUM_LANES
        distance_state = (state // NUM_LANES) % NUM_DISTANCE_STATES
        shuttle_lane = state // (NUM_LANES * NUM_DISTANCE_STATES)
        
        distance_names = ['Proche', 'Moyen', 'Loin']
        
        return {
            'shuttle_lane': shuttle_lane,
            'distance_state': distance_state,
            'distance_name': distance_names[distance_state],
            'obstacle_lane': obstacle_lane,
            'best_action': int(np.argmax(self.q_table[state])),
            'best_action_name': ACTION_NAMES[int(np.argmax(self.q_table[state]))],
            'q_values': self.q_table[state].tolist()
        }
    
    def save(self, filepath: str = None):
        """
        Sauvegarde la table Q.
        
        Args:
            filepath: Chemin du fichier (par défaut: Q_TABLE_FILE)
        """
        if filepath is None:
            filepath = Q_TABLE_FILE
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Sauvegarder la table Q et les métadonnées
        np.savez(
            filepath.replace('.npy', '.npz'),
            q_table=self.q_table,
            epsilon=self.epsilon,
            total_steps=self.total_steps,
            total_episodes=self.total_episodes
        )
        
        print(f"✅ Table Q sauvegardée: {filepath}")
    
    def load(self, filepath: str = None):
        """
        Charge une table Q sauvegardée.
        
        Args:
            filepath: Chemin du fichier (par défaut: Q_TABLE_FILE)
        """
        if filepath is None:
            filepath = Q_TABLE_FILE
        
        npz_path = filepath.replace('.npy', '.npz')
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            self.q_table = data['q_table']
            self.epsilon = float(data['epsilon'])
            self.total_steps = int(data['total_steps'])
            self.total_episodes = int(data['total_episodes'])
            print(f"✅ Table Q chargée: {npz_path}")
            print(f"   Épisodes: {self.total_episodes}, Epsilon: {self.epsilon:.4f}")
        else:
            raise FileNotFoundError(f"Fichier non trouvé: {npz_path}")
    
    def get_q_table_stats(self) -> Dict:
        """
        Retourne des statistiques sur la table Q.
        
        Returns:
            Dictionnaire de statistiques
        """
        return {
            'shape': self.q_table.shape,
            'min': float(np.min(self.q_table)),
            'max': float(np.max(self.q_table)),
            'mean': float(np.mean(self.q_table)),
            'std': float(np.std(self.q_table)),
            'non_zero': int(np.count_nonzero(self.q_table)),
            'sparsity': float(1 - np.count_nonzero(self.q_table) / self.q_table.size)
        }
    
    def print_policy(self):
        """Affiche la politique apprise de manière lisible."""
        print("\n" + "=" * 60)
        print("📋 POLITIQUE APPRISE")
        print("=" * 60)
        
        distance_names = ['Proche', 'Moyen', 'Loin']
        
        for shuttle_lane in range(NUM_LANES):
            print(f"\n🚐 Navette sur ligne {shuttle_lane + 1}:")
            print("-" * 40)
            
            for dist_state, dist_name in enumerate(distance_names):
                print(f"  📏 Distance {dist_name}:")
                
                for obs_lane in range(NUM_LANES):
                    state = (shuttle_lane * NUM_DISTANCE_STATES * NUM_LANES +
                            dist_state * NUM_LANES + obs_lane)
                    
                    best_action = int(np.argmax(self.q_table[state]))
                    q_values = self.q_table[state]
                    
                    action_symbol = ['↑', '●', '↓'][best_action]
                    
                    print(f"    Obstacle L{obs_lane + 1}: {action_symbol} {ACTION_NAMES[best_action]:10s} "
                          f"(Q: {q_values[0]:+.1f}, {q_values[1]:+.1f}, {q_values[2]:+.1f})")
