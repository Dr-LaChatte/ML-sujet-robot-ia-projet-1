"""
Script de génération du dataset.
Phase 2 du projet : Génération automatique du dataset à partir de la politique RL.
"""

import sys
import os
import argparse

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASET_SIZE, Q_TABLE_FILE, DATASET_FILE, RESULTS_DIR
from src.agents.q_learning_agent import QLearningAgent
from src.utils.dataset_generator import DatasetGenerator
from src.utils.visualization import plot_dataset_distribution


def generate_dataset(
    num_samples: int = DATASET_SIZE,
    agent_path: str = None,
    output_path: str = None,
    show_plots: bool = True
):
    """
    Génère le dataset à partir de la politique Q-learning entraînée.
    
    Args:
        num_samples: Nombre d'échantillons à générer
        agent_path: Chemin de l'agent Q-learning
        output_path: Chemin de sortie du dataset
        show_plots: Afficher les graphiques
    """
    print("=" * 60)
    print("📊 PHASE 2 : GÉNÉRATION DU DATASET")
    print("=" * 60)
    
    if agent_path is None:
        agent_path = Q_TABLE_FILE
    
    if output_path is None:
        output_path = DATASET_FILE
    
    # Charger l'agent entraîné
    print(f"\n📥 Chargement de l'agent Q-learning...")
    agent = QLearningAgent()
    
    try:
        agent.load(agent_path)
    except FileNotFoundError:
        print(f"\n❌ Erreur: Agent non trouvé à {agent_path}")
        print("   Veuillez d'abord exécuter train_rl.py pour entraîner l'agent.")
        return None, None
    
    print(f"\n📋 Configuration:")
    print(f"   Échantillons à générer: {num_samples}")
    print(f"   Agent: {agent_path}")
    print(f"   Sortie: {output_path}")
    
    # Créer le générateur
    generator = DatasetGenerator(agent)
    
    # Générer le dataset
    X, y = generator.generate(num_samples=num_samples)
    
    # Sauvegarder
    generator.save_dataset(X, y, output_path)
    
    # Afficher les graphiques
    if show_plots:
        print("\n📈 Génération des graphiques...")
        
        plot_dataset_distribution(
            X, y,
            save_path=os.path.join(RESULTS_DIR, "dataset_distribution.png"),
            show=True
        )
    
    print("\n✅ Génération du dataset terminée!")
    print(f"   Dataset: {output_path}")
    print(f"   Taille: {len(X)} échantillons")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Génération du dataset à partir de la politique Q-learning"
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=DATASET_SIZE,
        help=f"Nombre d'échantillons (défaut: {DATASET_SIZE})"
    )
    parser.add_argument(
        '--agent', '-a',
        type=str,
        default=None,
        help="Chemin de l'agent Q-learning"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Chemin de sortie du dataset"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    
    args = parser.parse_args()
    
    generate_dataset(
        num_samples=args.samples,
        agent_path=args.agent,
        output_path=args.output,
        show_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
