"""
Script principal - Exécute toutes les phases du projet.
"""

import sys
import os
import argparse
import time

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_rl import train_q_learning
from scripts.generate_dataset import generate_dataset
from scripts.train_ml import train_knn
from scripts.compare_agents import compare_agents


def run_all_phases(
    skip_training: bool = False,
    show_plots: bool = True
):
    """
    Exécute toutes les phases du projet.
    
    Args:
        skip_training: Passer l'entraînement si les modèles existent
        show_plots: Afficher les graphiques
    """
    print("\n" + "=" * 70)
    print("🚀 PROJET NAVETTE ROBOTIQUE ANTI-COLLISION")
    print("   Comparaison Reinforcement Learning vs Machine Learning Supervisé")
    print("=" * 70)
    
    start_time = time.time()
    
    # Phase 1 : Q-Learning
    print("\n\n" + "🔷" * 30)
    print("PHASE 1 : REINFORCEMENT LEARNING (Q-Learning)")
    print("🔷" * 30)
    
    rl_agent = train_q_learning(
        num_episodes=2000,
        show_plots=show_plots
    )
    
    if rl_agent is None:
        print("❌ Échec de l'entraînement RL. Arrêt.")
        return
    
    # Phase 2 : Génération du dataset
    print("\n\n" + "🔶" * 30)
    print("PHASE 2 : GÉNÉRATION DU DATASET")
    print("🔶" * 30)
    
    X, y = generate_dataset(
        num_samples=10000,
        show_plots=show_plots
    )
    
    if X is None:
        print("❌ Échec de la génération du dataset. Arrêt.")
        return
    
    # Phase 3 : k-NN
    print("\n\n" + "🔷" * 30)
    print("PHASE 3 : MACHINE LEARNING SUPERVISÉ (k-NN)")
    print("🔷" * 30)
    
    ml_agent = train_knn(
        find_optimal_k=True,
        show_plots=show_plots
    )
    
    if ml_agent is None:
        print("❌ Échec de l'entraînement ML. Arrêt.")
        return
    
    # Phase 4 : Comparaison
    print("\n\n" + "🔶" * 30)
    print("PHASE 4 : ANALYSE COMPARATIVE")
    print("🔶" * 30)
    
    comparison = compare_agents(
        num_episodes=100,
        analyze_stab=True,
        show_plots=show_plots
    )
    
    # Résumé final
    elapsed_time = time.time() - start_time
    
    print("\n\n" + "=" * 70)
    print("✅ PROJET TERMINÉ")
    print("=" * 70)
    print(f"\n   Temps total: {elapsed_time/60:.1f} minutes")
    print(f"\n   Fichiers générés:")
    print(f"   📁 models/q_table.npz     - Table Q entraînée")
    print(f"   📁 models/knn_model.pkl   - Modèle k-NN entraîné")
    print(f"   📁 data/dataset.csv       - Dataset généré")
    print(f"   📁 results/               - Graphiques et résultats")
    
    print(f"\n   Pour lancer la démonstration visuelle:")
    print(f"   python scripts/demo.py --agent rl")
    print(f"   python scripts/demo.py --agent ml")
    print(f"   python scripts/demo.py --agent manual")


def main():
    parser = argparse.ArgumentParser(
        description="Projet Navette Robotique - Exécution complète"
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help="Passer l'entraînement si les modèles existent"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    
    args = parser.parse_args()
    
    run_all_phases(
        skip_training=args.skip_training,
        show_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
