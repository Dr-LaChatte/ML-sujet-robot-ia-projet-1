"""
Agent k-NN pour le ML supervisé.
Apprend à imiter la politique Q-learning à partir du dataset généré.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os
import sys
from typing import Tuple, Dict, Optional

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    KNN_NEIGHBORS, KNN_WEIGHTS,
    TEST_SIZE, VALIDATION_SIZE,
    KNN_MODEL_FILE, ACTION_NAMES, NUM_ACTIONS
)


class KNNAgent:
    """
    Agent k-NN pour l'apprentissage supervisé.
    
    Imite la politique apprise par Q-learning en classifiant
    les états pour prédire les actions.
    """
    
    def __init__(
        self,
        n_neighbors: int = KNN_NEIGHBORS,
        weights: str = KNN_WEIGHTS
    ):
        """
        Initialise l'agent k-NN.
        
        Args:
            n_neighbors: Nombre de voisins
            weights: Pondération ('uniform' ou 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='auto',
            metric='euclidean'
        )
        
        # Données d'entraînement
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Métriques
        self.metrics = {}
        self.is_trained = False
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE,
        random_state: int = 42
    ) -> Dict:
        """
        Prépare les données en séparant train/val/test.
        
        Args:
            X: Features (états)
            y: Labels (actions)
            test_size: Proportion pour le test
            val_size: Proportion pour la validation
            random_state: Graine aléatoire
            
        Returns:
            Dictionnaire avec les tailles des ensembles
        """
        # Première séparation : train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Deuxième séparation : train vs val
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        data_info = {
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test),
            'total_size': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'n_classes': len(np.unique(y)),
            'class_distribution': {
                int(c): int(np.sum(y == c)) for c in np.unique(y)
            }
        }
        
        print("\n📊 Préparation des données:")
        print(f"   Train: {data_info['train_size']} ({100*(1-test_size-val_size):.0f}%)")
        print(f"   Validation: {data_info['val_size']} ({100*val_size:.0f}%)")
        print(f"   Test: {data_info['test_size']} ({100*test_size:.0f}%)")
        print(f"   Classes: {data_info['class_distribution']}")
        
        return data_info
    
    def train(self) -> Dict:
        """
        Entraîne le modèle k-NN.
        
        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        if self.X_train is None:
            raise ValueError("Données non préparées. Appeler prepare_data() d'abord.")
        
        print(f"\n🎓 Entraînement k-NN (k={self.n_neighbors}, weights={self.weights})...")
        
        # Entraînement
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        # Évaluation sur validation
        y_val_pred = self.model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        
        # Validation croisée
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        
        self.metrics['train'] = {
            'val_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        print(f"   ✅ Accuracy validation: {val_accuracy:.4f}")
        print(f"   ✅ Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return self.metrics['train']
    
    def evaluate(self) -> Dict:
        """
        Évalue le modèle sur l'ensemble de test.
        
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if not self.is_trained:
            raise ValueError("Modèle non entraîné. Appeler train() d'abord.")
        
        print("\n📈 Évaluation sur l'ensemble de test...")
        
        # Prédictions
        y_pred = self.model.predict(self.X_test)
        
        # Métriques
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(
            self.y_test, y_pred,
            target_names=[ACTION_NAMES[i] for i in range(NUM_ACTIONS)],
            output_dict=True
        )
        
        self.metrics['test'] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        print(f"   ✅ Accuracy test: {accuracy:.4f}")
        print("\n📊 Matrice de confusion:")
        print(f"   {'':12s} Prédit →")
        print(f"   {'Réel ↓':12s} {'Monter':>8s} {'Rester':>8s} {'Descendre':>10s}")
        for i, row in enumerate(conf_matrix):
            print(f"   {ACTION_NAMES[i]:12s} {row[0]:>8d} {row[1]:>8d} {row[2]:>10d}")
        
        print("\n📋 Rapport de classification:")
        for action_id, action_name in ACTION_NAMES.items():
            if action_name in class_report:
                stats = class_report[action_name]
                print(f"   {action_name:12s} - Précision: {stats['precision']:.3f}, "
                      f"Rappel: {stats['recall']:.3f}, F1: {stats['f1-score']:.3f}")
        
        return self.metrics['test']
    
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choisit une action pour un état donné.
        
        Args:
            state: État (array de features)
            
        Returns:
            Action prédite
        """
        if not self.is_trained:
            raise ValueError("Modèle non entraîné.")
        
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        return int(self.model.predict(state)[0])
    
    def choose_action_from_components(
        self,
        shuttle_lane: int,
        distance_state: int,
        obstacle_lane: int
    ) -> int:
        """
        Choisit une action à partir des composantes d'état.
        
        Args:
            shuttle_lane: Ligne de la navette
            distance_state: Distance discrétisée (0=proche, 1=moyen, 2=loin)
            obstacle_lane: Ligne de l'obstacle
            
        Returns:
            Action prédite
        """
        state = np.array([[shuttle_lane, distance_state, obstacle_lane]])
        return self.choose_action(state)
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités pour chaque action (basé sur les voisins).
        
        Args:
            state: État
            
        Returns:
            Array de probabilités pour chaque action
        """
        if not self.is_trained:
            raise ValueError("Modèle non entraîné.")
        
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Trouver les k voisins
        distances, indices = self.model.kneighbors(state)
        neighbor_labels = self.y_train[indices[0]]
        
        # Calculer les probabilités
        probs = np.zeros(NUM_ACTIONS)
        for label in neighbor_labels:
            probs[label] += 1
        probs /= len(neighbor_labels)
        
        return probs
    
    def save(self, filepath: str = None):
        """
        Sauvegarde le modèle k-NN.
        
        Args:
            filepath: Chemin du fichier
        """
        if filepath is None:
            filepath = KNN_MODEL_FILE
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights
            }, f)
        
        print(f"✅ Modèle k-NN sauvegardé: {filepath}")
    
    def load(self, filepath: str = None):
        """
        Charge un modèle k-NN sauvegardé.
        
        Args:
            filepath: Chemin du fichier
        """
        if filepath is None:
            filepath = KNN_MODEL_FILE
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.metrics = data['metrics']
        self.n_neighbors = data['n_neighbors']
        self.weights = data['weights']
        self.is_trained = True
        
        print(f"✅ Modèle k-NN chargé: {filepath}")
        if 'test' in self.metrics:
            print(f"   Accuracy: {self.metrics['test']['accuracy']:.4f}")


def find_best_k(X: np.ndarray, y: np.ndarray, k_range: range = range(1, 21)) -> Dict:
    """
    Trouve le meilleur k par validation croisée.
    
    Args:
        X: Features
        y: Labels
        k_range: Plage de valeurs de k à tester
        
    Returns:
        Dictionnaire avec les résultats
    """
    print("\n🔍 Recherche du meilleur k...")
    
    results = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X, y, cv=5)
        results.append({
            'k': k,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        })
        print(f"   k={k:2d}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    best = max(results, key=lambda x: x['mean_score'])
    print(f"\n   🏆 Meilleur k = {best['k']} (accuracy = {best['mean_score']:.4f})")
    
    return {
        'best_k': best['k'],
        'best_score': best['mean_score'],
        'all_results': results
    }
