"""
Script de comparaison des agents RL et ML.
Phase 4 du projet : Analyse comparative.
"""

import sys
import os
import argparse
import json

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    Q_TABLE_FILE, KNN_MODEL_FILE, RESULTS_DIR, EVAL_EPISODES
)
from src.agents.q_learning_agent import QLearningAgent
from src.agents.knn_agent import KNNAgent
from src.utils.metrics import AgentEvaluator, analyze_stability
from src.utils.visualization import plot_comparison_results


def compare_agents(
    rl_agent_path: str = None,
    ml_agent_path: str = None,
    num_episodes: int = EVAL_EPISODES,
    analyze_stab: bool = True,
    show_plots: bool = True,
    save_results: bool = True
):
    """
    Compare les performances des agents RL et ML.
    
    Args:
        rl_agent_path: Chemin de l'agent Q-learning
        ml_agent_path: Chemin de l'agent k-NN
        num_episodes: Nombre d'épisodes d'évaluation
        analyze_stab: Analyser la stabilité
        show_plots: Afficher les graphiques
        save_results: Sauvegarder les résultats
    """
    print("=" * 60)
    print("🔬 PHASE 4 : COMPARAISON RL vs ML")
    print("=" * 60)
    
    if rl_agent_path is None:
        rl_agent_path = Q_TABLE_FILE
    
    if ml_agent_path is None:
        ml_agent_path = KNN_MODEL_FILE
    
    # Charger les agents
    print(f"\n📥 Chargement des agents...")
    
    rl_agent = QLearningAgent()
    ml_agent = KNNAgent()
    
    try:
        rl_agent.load(rl_agent_path)
    except FileNotFoundError:
        print(f"\n❌ Erreur: Agent RL non trouvé à {rl_agent_path}")
        print("   Veuillez d'abord exécuter train_rl.py")
        return None
    
    try:
        ml_agent.load(ml_agent_path)
    except FileNotFoundError:
        print(f"\n❌ Erreur: Agent ML non trouvé à {ml_agent_path}")
        print("   Veuillez d'abord exécuter train_ml.py")
        return None
    
    print(f"   ✅ Agent RL chargé")
    print(f"   ✅ Agent ML chargé (k={ml_agent.n_neighbors})")
    
    print(f"\n📋 Configuration:")
    print(f"   Épisodes d'évaluation: {num_episodes}")
    
    # Évaluer et comparer
    evaluator = AgentEvaluator()
    comparison = evaluator.compare_agents(rl_agent, ml_agent, num_episodes)
    
    # Analyse de stabilité
    if analyze_stab:
        print("\n" + "=" * 60)
        print("📈 ANALYSE DE STABILITÉ")
        print("=" * 60)
        
        rl_stability = analyze_stability(rl_agent, 'rl', num_runs=5, episodes_per_run=50)
        ml_stability = analyze_stability(ml_agent, 'ml', num_runs=5, episodes_per_run=50)
        
        comparison['rl_stability'] = rl_stability
        comparison['ml_stability'] = ml_stability
        
        print("\n📊 Comparaison de stabilité:")
        print(f"   RL - CV score: {rl_stability['cv_score']:.2%}")
        print(f"   ML - CV score: {ml_stability['cv_score']:.2%}")
        
        if rl_stability['cv_score'] < ml_stability['cv_score']:
            print(f"   → RL est plus stable (moins de variation)")
        else:
            print(f"   → ML est plus stable (moins de variation)")
    
    # Conclusions
    print("\n" + "=" * 60)
    print("📝 CONCLUSIONS")
    print("=" * 60)
    
    rl_metrics = comparison['rl']
    ml_metrics = comparison['ml']
    
    print("\n🎯 Avantages du Reinforcement Learning (Q-Learning):")
    print("   • Apprend directement par interaction avec l'environnement")
    print("   • S'adapte naturellement aux changements de l'environnement")
    print("   • Ne nécessite pas de données étiquetées")
    print("   • Optimise directement la politique de décision")
    
    print("\n📊 Avantages du Machine Learning Supervisé (k-NN):")
    print("   • Entraînement très rapide une fois le dataset disponible")
    print("   • Prédictions rapides et simples")
    print("   • Facilement interprétable")
    print("   • Pas de phase d'exploration (stabilité immédiate)")
    
    print("\n⚠️ Limites:")
    print("   • RL: Temps d'entraînement long, exploration peut être dangereuse")
    print("   • ML: Dépend de la qualité du dataset, ne s'adapte pas aux changements")
    
    print("\n🏆 Recommandation:")
    if rl_metrics['avoidance_rate'] >= ml_metrics['avoidance_rate']:
        print("   Pour un environnement dynamique et évolutif: RL (Q-Learning)")
    print("   Pour un déploiement rapide avec environnement stable: ML (k-NN)")
    
    # Sauvegarder les résultats
    if save_results:
        # Préparer les résultats pour JSON (enlever les listes numpy)
        results_to_save = {
            'num_episodes': num_episodes,
            'rl': {
                'avoidance_rate': rl_metrics['avoidance_rate'],
                'collision_rate': rl_metrics['collision_rate'],
                'avg_reward': rl_metrics['avg_reward'],
                'avg_episode_length': rl_metrics['avg_episode_length'],
                'total_score': rl_metrics['total_score']
            },
            'ml': {
                'avoidance_rate': ml_metrics['avoidance_rate'],
                'collision_rate': ml_metrics['collision_rate'],
                'avg_reward': ml_metrics['avg_reward'],
                'avg_episode_length': ml_metrics['avg_episode_length'],
                'total_score': ml_metrics['total_score']
            }
        }
        
        results_path = os.path.join(RESULTS_DIR, "comparison_results.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {results_path}")
    
    # Graphiques
    if show_plots:
        print("\n📈 Génération des graphiques...")
        
        plot_comparison_results(
            comparison,
            save_path=os.path.join(RESULTS_DIR, "comparison_results.png"),
            show=True
        )
    
    print("\n✅ Comparaison terminée!")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison des agents RL et ML"
    )
    parser.add_argument(
        '--rl-agent',
        type=str,
        default=None,
        help="Chemin de l'agent Q-learning"
    )
    parser.add_argument(
        '--ml-agent',
        type=str,
        default=None,
        help="Chemin de l'agent k-NN"
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=EVAL_EPISODES,
        help=f"Nombre d'épisodes d'évaluation (défaut: {EVAL_EPISODES})"
    )
    parser.add_argument(
        '--no-stability',
        action='store_true',
        help="Ne pas analyser la stabilité"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Ne pas sauvegarder les résultats"
    )
    
    args = parser.parse_args()
    
    compare_agents(
        rl_agent_path=args.rl_agent,
        ml_agent_path=args.ml_agent,
        num_episodes=args.episodes,
        analyze_stab=not args.no_stability,
        show_plots=not args.no_plots,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
