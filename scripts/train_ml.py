"""
Script d'entraînement du modèle k-NN.
Phase 3 du projet : Machine Learning supervisé.
"""

import sys
import os
import argparse
import numpy as np

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATASET_FILE, KNN_MODEL_FILE, RESULTS_DIR,
    KNN_NEIGHBORS, KNN_WEIGHTS
)
from src.agents.knn_agent import KNNAgent, find_best_k
from src.utils.dataset_generator import DatasetGenerator
from src.utils.visualization import plot_confusion_matrix


def train_knn(
    dataset_path: str = None,
    output_path: str = None,
    k: int = None,
    find_optimal_k: bool = False,
    show_plots: bool = True
):
    """
    Entraîne le modèle k-NN sur le dataset généré.
    
    Args:
        dataset_path: Chemin du dataset
        output_path: Chemin de sauvegarde du modèle
        k: Nombre de voisins (si None, utilise la valeur par défaut)
        find_optimal_k: Rechercher le k optimal
        show_plots: Afficher les graphiques
    """
    print("=" * 60)
    print("🎓 PHASE 3 : ENTRAÎNEMENT k-NN")
    print("=" * 60)
    
    if dataset_path is None:
        dataset_path = DATASET_FILE
    
    if output_path is None:
        output_path = KNN_MODEL_FILE
    
    # Charger le dataset
    print(f"\n📥 Chargement du dataset...")
    
    try:
        X, y = DatasetGenerator.load_dataset(dataset_path)
    except FileNotFoundError:
        print(f"\n❌ Erreur: Dataset non trouvé à {dataset_path}")
        print("   Veuillez d'abord exécuter generate_dataset.py")
        return None
    
    print(f"\n📋 Dataset:")
    print(f"   Taille: {len(X)} échantillons")
    print(f"   Features: {X.shape[1]} (ligne navette, distance, ligne obstacle)")
    print(f"   Classes: {len(np.unique(y))} actions")
    
    # Trouver le k optimal si demandé
    if find_optimal_k:
        print("\n🔍 Recherche du k optimal...")
        best_k_result = find_best_k(X, y, k_range=range(1, 21))
        k = best_k_result['best_k']
    elif k is None:
        k = KNN_NEIGHBORS
    
    print(f"\n📋 Configuration k-NN:")
    print(f"   k (voisins): {k}")
    print(f"   Pondération: {KNN_WEIGHTS}")
    
    # Créer et entraîner l'agent
    agent = KNNAgent(n_neighbors=k, weights=KNN_WEIGHTS)
    
    # Préparer les données
    data_info = agent.prepare_data(X, y)
    
    # Entraîner
    train_metrics = agent.train()
    
    # Évaluer
    test_metrics = agent.evaluate()
    
    # Sauvegarder
    agent.save(output_path)
    
    # Afficher les graphiques
    if show_plots:
        print("\n📈 Génération des graphiques...")
        
        conf_matrix = np.array(test_metrics['confusion_matrix'])
        plot_confusion_matrix(
            conf_matrix,
            save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
            show=True
        )
    
    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"\n   Accuracy test: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   Cross-validation: {train_metrics['cv_mean']:.4f} (+/- {train_metrics['cv_std']*2:.4f})")
    print(f"\n   Modèle sauvegardé: {output_path}")
    
    print("\n✅ Entraînement k-NN terminé!")
    
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle k-NN"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help="Chemin du dataset"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Chemin de sauvegarde du modèle"
    )
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=None,
        help=f"Nombre de voisins (défaut: {KNN_NEIGHBORS})"
    )
    parser.add_argument(
        '--find-optimal-k',
        action='store_true',
        help="Rechercher le k optimal par validation croisée"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    
    args = parser.parse_args()
    
    train_knn(
        dataset_path=args.dataset,
        output_path=args.output,
        k=args.k,
        find_optimal_k=args.find_optimal_k,
        show_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
