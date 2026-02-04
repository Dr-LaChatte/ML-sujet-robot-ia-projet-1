"""
Outils de visualisation pour le projet.
Courbes d'apprentissage, politique, comparaisons, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import sys

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    RESULTS_DIR, NUM_LANES, NUM_DISTANCE_STATES,
    ACTION_NAMES, NUM_ACTIONS
)


def plot_training_curves(
    history: Dict,
    save_path: str = None,
    show: bool = True
):
    """
    Trace les courbes d'apprentissage du Q-learning.
    
    Args:
        history: Historique d'entraînement de l'agent
        save_path: Chemin de sauvegarde (optionnel)
        show: Afficher le graphique
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = range(1, len(history['episode_rewards']) + 1)
    
    # 1. Récompenses par épisode
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['episode_rewards'], alpha=0.3, color='blue')
    # Moyenne mobile
    window = min(50, len(history['episode_rewards']) // 10 + 1)
    if window > 1:
        rewards_smooth = np.convolve(
            history['episode_rewards'],
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(range(window, len(history['episode_rewards']) + 1),
                rewards_smooth, color='blue', linewidth=2, label='Moyenne mobile')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('📈 Récompenses par épisode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Durée des épisodes
    ax2 = axes[0, 1]
    ax2.plot(episodes, history['episode_lengths'], alpha=0.3, color='green')
    if window > 1:
        lengths_smooth = np.convolve(
            history['episode_lengths'],
            np.ones(window) / window,
            mode='valid'
        )
        ax2.plot(range(window, len(history['episode_lengths']) + 1),
                lengths_smooth, color='green', linewidth=2, label='Moyenne mobile')
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Nombre de pas')
    ax2.set_title('⏱️ Durée des épisodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['epsilon_values'], color='orange', linewidth=2)
    ax3.set_xlabel('Épisode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('🎲 Décroissance de Epsilon')
    ax3.grid(True, alpha=0.3)
    
    # 4. Évitements vs Collisions
    ax4 = axes[1, 1]
    ax4.plot(episodes, np.cumsum(history['avoidances']), 
             color='green', linewidth=2, label='Évitements (cumulés)')
    ax4.plot(episodes, np.cumsum(history['collisions']),
             color='red', linewidth=2, label='Collisions (cumulées)')
    ax4.set_xlabel('Épisode')
    ax4.set_ylabel('Nombre cumulé')
    ax4.set_title('🎯 Évitements vs Collisions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_q_table_heatmap(
    q_table: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Visualise la table Q sous forme de heatmap.
    
    Args:
        q_table: Table Q (états x actions)
        save_path: Chemin de sauvegarde
        show: Afficher le graphique
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    action_labels = ['Monter', 'Rester', 'Descendre']
    distance_labels = ['Proche', 'Moyen', 'Loin']
    
    for action_id, ax in enumerate(axes):
        # Extraire les valeurs Q pour cette action
        q_values = q_table[:, action_id].reshape(NUM_LANES, NUM_DISTANCE_STATES * NUM_LANES)
        
        im = ax.imshow(q_values, cmap='RdYlGn', aspect='auto')
        ax.set_title(f'Action: {action_labels[action_id]}')
        ax.set_xlabel('Distance × Ligne obstacle')
        ax.set_ylabel('Ligne navette')
        ax.set_yticks(range(NUM_LANES))
        ax.set_yticklabels([f'L{i+1}' for i in range(NUM_LANES)])
        
        plt.colorbar(im, ax=ax, label='Valeur Q')
    
    plt.suptitle('🗺️ Table Q - Valeurs par action', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Heatmap sauvegardée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_policy_visualization(
    q_table: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Visualise la politique apprise.
    
    Args:
        q_table: Table Q
        save_path: Chemin de sauvegarde
        show: Afficher le graphique
    """
    fig, axes = plt.subplots(1, NUM_DISTANCE_STATES, figsize=(15, 6))
    
    distance_labels = ['Proche', 'Moyen', 'Loin']
    action_colors = {0: 'green', 1: 'blue', 2: 'orange'}
    action_markers = {0: '↑', 1: '●', 2: '↓'}
    
    for dist_idx, ax in enumerate(axes):
        for shuttle_lane in range(NUM_LANES):
            for obs_lane in range(NUM_LANES):
                state = (shuttle_lane * NUM_DISTANCE_STATES * NUM_LANES +
                        dist_idx * NUM_LANES + obs_lane)
                
                best_action = int(np.argmax(q_table[state]))
                
                ax.scatter(obs_lane, shuttle_lane,
                          c=action_colors[best_action],
                          s=500, marker='s', alpha=0.7)
                ax.text(obs_lane, shuttle_lane, action_markers[best_action],
                       ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_xlim(-0.5, NUM_LANES - 0.5)
        ax.set_ylim(-0.5, NUM_LANES - 0.5)
        ax.set_xlabel('Ligne obstacle')
        ax.set_ylabel('Ligne navette')
        ax.set_title(f'Distance: {distance_labels[dist_idx]}')
        ax.set_xticks(range(NUM_LANES))
        ax.set_yticks(range(NUM_LANES))
        ax.set_xticklabels([f'L{i+1}' for i in range(NUM_LANES)])
        ax.set_yticklabels([f'L{i+1}' for i in range(NUM_LANES)])
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='↑ Monter'),
        Patch(facecolor='blue', label='● Rester'),
        Patch(facecolor='orange', label='↓ Descendre')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('📋 Politique apprise par Q-Learning', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Politique visualisée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Affiche la matrice de confusion.
    
    Args:
        conf_matrix: Matrice de confusion
        save_path: Chemin de sauvegarde
        show: Afficher le graphique
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(conf_matrix, cmap='Blues')
    
    # Labels
    labels = ['Monter', 'Rester', 'Descendre']
    ax.set_xticks(range(NUM_ACTIONS))
    ax.set_yticks(range(NUM_ACTIONS))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('📊 Matrice de Confusion - k-NN')
    
    # Valeurs dans les cellules
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
            ax.text(j, i, str(conf_matrix[i, j]),
                   ha='center', va='center', color=color, fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_results(
    comparison_results: Dict,
    save_path: str = None,
    show: bool = True
):
    """
    Affiche la comparaison entre agents RL et ML.
    
    Args:
        comparison_results: Résultats de la comparaison
        save_path: Chemin de sauvegarde
        show: Afficher le graphique
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    agents = ['RL (Q-Learning)', 'ML (k-NN)']
    colors = ['#3498db', '#e74c3c']
    
    # 1. Taux d'évitement
    ax1 = axes[0, 0]
    avoid_rates = [
        comparison_results['rl']['avoidance_rate'],
        comparison_results['ml']['avoidance_rate']
    ]
    bars1 = ax1.bar(agents, avoid_rates, color=colors)
    ax1.set_ylabel('Taux (%)')
    ax1.set_title('✅ Taux d\'évitement')
    ax1.set_ylim(0, 100)
    for bar, rate in zip(bars1, avoid_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=12)
    
    # 2. Taux de collision
    ax2 = axes[0, 1]
    coll_rates = [
        comparison_results['rl']['collision_rate'],
        comparison_results['ml']['collision_rate']
    ]
    bars2 = ax2.bar(agents, coll_rates, color=colors)
    ax2.set_ylabel('Taux (%)')
    ax2.set_title('💥 Taux de collision')
    ax2.set_ylim(0, max(coll_rates) * 1.3 + 1)
    for bar, rate in zip(bars2, coll_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=12)
    
    # 3. Durée moyenne des épisodes
    ax3 = axes[1, 0]
    avg_lengths = [
        comparison_results['rl']['avg_episode_length'],
        comparison_results['ml']['avg_episode_length']
    ]
    bars3 = ax3.bar(agents, avg_lengths, color=colors)
    ax3.set_ylabel('Nombre de pas')
    ax3.set_title('⏱️ Durée moyenne des épisodes')
    for bar, length in zip(bars3, avg_lengths):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{length:.0f}', ha='center', fontsize=12)
    
    # 4. Score global
    ax4 = axes[1, 1]
    scores = [
        comparison_results['rl']['total_score'],
        comparison_results['ml']['total_score']
    ]
    bars4 = ax4.bar(agents, scores, color=colors)
    ax4.set_ylabel('Score')
    ax4.set_title('🏆 Score total (récompenses)')
    for bar, score in zip(bars4, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(score)*0.02,
                f'{score:.0f}', ha='center', fontsize=12)
    
    plt.suptitle('📊 Comparaison RL vs ML Supervisé', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparaison sauvegardée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_distribution(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Visualise la distribution du dataset.
    
    Args:
        X: Features
        y: Labels
        save_path: Chemin de sauvegarde
        show: Afficher le graphique
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribution des actions
    ax1 = axes[0, 0]
    action_counts = [np.sum(y == i) for i in range(NUM_ACTIONS)]
    colors = ['green', 'blue', 'orange']
    bars = ax1.bar(['Monter', 'Rester', 'Descendre'], action_counts, color=colors)
    ax1.set_ylabel('Nombre d\'échantillons')
    ax1.set_title('📊 Distribution des actions')
    for bar, count in zip(bars, action_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', fontsize=10)
    
    # 2. Distribution par ligne navette
    ax2 = axes[0, 1]
    lane_counts = [np.sum(X[:, 0] == i) for i in range(NUM_LANES)]
    ax2.bar([f'L{i+1}' for i in range(NUM_LANES)], lane_counts, color='#3498db')
    ax2.set_ylabel('Nombre d\'échantillons')
    ax2.set_title('🚐 Distribution par ligne navette')
    
    # 3. Distribution par distance
    ax3 = axes[1, 0]
    dist_counts = [np.sum(X[:, 1] == i) for i in range(NUM_DISTANCE_STATES)]
    ax3.bar(['Proche', 'Moyen', 'Loin'], dist_counts, color=['red', 'yellow', 'green'])
    ax3.set_ylabel('Nombre d\'échantillons')
    ax3.set_title('📏 Distribution par distance obstacle')
    
    # 4. Heatmap lignes navette vs obstacle
    ax4 = axes[1, 1]
    heatmap_data = np.zeros((NUM_LANES, NUM_LANES))
    for i in range(len(X)):
        heatmap_data[int(X[i, 0]), int(X[i, 2])] += 1
    im = ax4.imshow(heatmap_data, cmap='YlOrRd')
    ax4.set_xlabel('Ligne obstacle')
    ax4.set_ylabel('Ligne navette')
    ax4.set_xticks(range(NUM_LANES))
    ax4.set_yticks(range(NUM_LANES))
    ax4.set_xticklabels([f'L{i+1}' for i in range(NUM_LANES)])
    ax4.set_yticklabels([f'L{i+1}' for i in range(NUM_LANES)])
    ax4.set_title('🔥 Fréquence des configurations')
    plt.colorbar(im, ax=ax4)
    
    plt.suptitle('📈 Distribution du Dataset', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Distribution sauvegardée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
