"""
Générateur de dataset à partir de la politique Q-learning.
Enregistre les états et actions pour l'apprentissage supervisé.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    DATASET_SIZE, DATASET_FILE, ACTION_NAMES,
    MAX_STEPS_PER_EPISODE
)
from src.environment.warehouse_env import WarehouseEnvFast
from src.agents.q_learning_agent import QLearningAgent


class DatasetGenerator:
    """
    Génère un dataset en faisant exécuter la politique RL apprise.
    
    Le dataset contient des paires (état, action) pour entraîner
    un modèle de ML supervisé.
    """
    
    def __init__(self, agent: QLearningAgent):
        """
        Initialise le générateur.
        
        Args:
            agent: Agent Q-learning entraîné
        """
        self.agent = agent
        self.env = WarehouseEnvFast()
        
        # Données collectées
        self.states = []
        self.actions = []
        self.state_components = []  # (shuttle_lane, distance_state, obstacle_lane)
        
        # Statistiques
        self.stats = {
            'total_samples': 0,
            'episodes': 0,
            'total_steps': 0,
            'collisions': 0,
            'avoidances': 0,
            'action_distribution': {0: 0, 1: 0, 2: 0}
        }
    
    def generate(
        self,
        num_samples: int = DATASET_SIZE,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère le dataset.
        
        Args:
            num_samples: Nombre d'échantillons à collecter
            max_steps: Nombre maximum de pas par épisode
            verbose: Afficher la progression
            
        Returns:
            (X, y) - Features et labels
        """
        print(f"\n🎲 Génération du dataset ({num_samples} échantillons)...")
        
        self.states = []
        self.actions = []
        self.state_components = []
        
        pbar = tqdm(total=num_samples, disable=not verbose, desc="Collecte")
        
        while len(self.states) < num_samples:
            state, _ = self.env.reset()
            self.stats['episodes'] += 1
            
            for step in range(max_steps):
                # Obtenir les composantes de l'état
                shuttle_lane, distance_state, obstacle_lane = self.env.get_state_components()
                
                # Choisir l'action (exploitation pure, pas d'exploration)
                action = self.agent.choose_action(state, training=False)
                
                # Enregistrer
                self.states.append(state)
                self.actions.append(action)
                self.state_components.append((shuttle_lane, distance_state, obstacle_lane))
                self.stats['action_distribution'][action] += 1
                
                pbar.update(1)
                
                if len(self.states) >= num_samples:
                    break
                
                # Exécuter l'action
                next_state, reward, terminated, _, _ = self.env.step(action)
                self.stats['total_steps'] += 1
                
                if reward > 0:
                    self.stats['avoidances'] += 1
                
                if terminated:
                    self.stats['collisions'] += 1
                    break
                
                state = next_state
        
        pbar.close()
        
        self.stats['total_samples'] = len(self.states)
        
        # Convertir en arrays numpy
        X = np.array(self.state_components)
        y = np.array(self.actions)
        
        self._print_stats()
        
        return X, y
    
    def _print_stats(self):
        """Affiche les statistiques de génération."""
        print("\n📊 Statistiques de génération:")
        print(f"   Échantillons: {self.stats['total_samples']}")
        print(f"   Épisodes: {self.stats['episodes']}")
        print(f"   Collisions: {self.stats['collisions']}")
        print(f"   Évitements: {self.stats['avoidances']}")
        
        print("\n   Distribution des actions:")
        total = sum(self.stats['action_distribution'].values())
        for action_id, count in self.stats['action_distribution'].items():
            pct = 100 * count / total if total > 0 else 0
            print(f"   {ACTION_NAMES[action_id]:12s}: {count:5d} ({pct:5.1f}%)")
    
    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        filepath: str = None
    ):
        """
        Sauvegarde le dataset au format CSV.
        
        Args:
            X: Features
            y: Labels
            filepath: Chemin du fichier
        """
        if filepath is None:
            filepath = DATASET_FILE
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Créer un DataFrame
        df = pd.DataFrame({
            'shuttle_lane': X[:, 0],
            'distance_state': X[:, 1],
            'obstacle_lane': X[:, 2],
            'action': y
        })
        
        # Ajouter les noms lisibles
        distance_names = {0: 'proche', 1: 'moyen', 2: 'loin'}
        df['distance_name'] = df['distance_state'].map(distance_names)
        df['action_name'] = df['action'].map(ACTION_NAMES)
        
        # Sauvegarder
        df.to_csv(filepath, index=False)
        
        print(f"\n✅ Dataset sauvegardé: {filepath}")
        print(f"   Taille: {len(df)} échantillons")
    
    @staticmethod
    def load_dataset(filepath: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge un dataset depuis un fichier CSV.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            (X, y) - Features et labels
        """
        if filepath is None:
            filepath = DATASET_FILE
        
        df = pd.read_csv(filepath)
        
        X = df[['shuttle_lane', 'distance_state', 'obstacle_lane']].values
        y = df['action'].values
        
        print(f"✅ Dataset chargé: {filepath}")
        print(f"   Taille: {len(df)} échantillons")
        
        return X, y


def generate_dataset_from_trained_agent(
    agent_path: str = None,
    num_samples: int = DATASET_SIZE,
    save_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fonction utilitaire pour générer un dataset.
    
    Args:
        agent_path: Chemin de l'agent Q-learning sauvegardé
        num_samples: Nombre d'échantillons
        save_path: Chemin de sauvegarde du dataset
        
    Returns:
        (X, y) - Features et labels
    """
    # Charger l'agent
    agent = QLearningAgent()
    agent.load(agent_path)
    
    # Générer le dataset
    generator = DatasetGenerator(agent)
    X, y = generator.generate(num_samples=num_samples)
    
    # Sauvegarder
    generator.save_dataset(X, y, save_path)
    
    return X, y
