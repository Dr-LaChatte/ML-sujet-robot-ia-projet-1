"""
Script d'entraînement de l'agent Q-Learning.
Phase 1 du projet : Apprentissage par renforcement.
"""

import sys
import os
import argparse
from tqdm import tqdm

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    NUM_EPISODES, MAX_STEPS_PER_EPISODE,
    Q_TABLE_FILE, RESULTS_DIR
)
from src.environment.warehouse_env import WarehouseEnvFast, WarehouseEnv
from src.agents.q_learning_agent import QLearningAgent
from src.utils.visualization import (
    plot_training_curves,
    plot_policy_visualization,
    plot_q_table_heatmap
)


def train_q_learning(
    num_episodes: int = NUM_EPISODES,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    render: bool = False,
    save_model: bool = True,
    show_plots: bool = True
):
    """
    Entraîne l'agent Q-Learning.
    
    Args:
        num_episodes: Nombre d'épisodes d'entraînement
        max_steps: Nombre maximum de pas par épisode
        render: Afficher la simulation pendant l'entraînement
        save_model: Sauvegarder le modèle
        show_plots: Afficher les graphiques
    """
    print("=" * 60)
    print("🎮 PHASE 1 : ENTRAÎNEMENT Q-LEARNING")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   Épisodes: {num_episodes}")
    print(f"   Max pas/épisode: {max_steps}")
    
    # Créer l'environnement et l'agent
    if render:
        env = WarehouseEnv(render_mode='human')
    else:
        env = WarehouseEnvFast()
    
    agent = QLearningAgent(num_states=env.num_states, num_actions=env.num_actions)
    
    print(f"   États: {agent.num_states}")
    print(f"   Actions: {agent.num_actions}")
    print(f"   Learning rate: {agent.learning_rate}")
    print(f"   Discount factor: {agent.discount_factor}")
    print(f"   Epsilon: {agent.epsilon} → {agent.epsilon_end}")
    
    # Boucle d'entraînement
    print(f"\n🚀 Début de l'entraînement...")
    
    best_reward = float('-inf')
    
    for episode in tqdm(range(num_episodes), desc="Entraînement"):
        state, _ = env.reset()
        total_reward = 0
        collisions = 0
        avoidances = 0
        
        for step in range(max_steps):
            # Choisir une action
            action = agent.choose_action(state, training=True)
            
            # Exécuter l'action
            next_state, reward, terminated, _, _ = env.step(action)
            
            # Apprendre
            agent.learn(state, action, reward, next_state, terminated)
            
            total_reward += reward
            
            if reward > 0:
                avoidances += 1
            
            if terminated:
                collisions = 1
                break
            
            state = next_state
            
            # Rendu si activé
            if render:
                continue_rendering = env.render()
                if not continue_rendering:
                    print("\n❌ Fenêtre fermée, arrêt de l'entraînement.")
                    return
        
        # Enregistrer l'épisode
        agent.record_episode(total_reward, step + 1, collisions, avoidances)
        
        # Décroître epsilon
        agent.decay_epsilon()
        
        # Afficher la progression
        if (episode + 1) % (num_episodes // 10) == 0:
            recent_rewards = agent.training_history['episode_rewards'][-100:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
            
            tqdm.write(f"   Épisode {episode+1}: Récompense moy. = {avg_reward:.1f}, "
                      f"Epsilon = {agent.epsilon:.3f}")
    
    # Fermer l'environnement
    if render:
        env.close()
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("=" * 60)
    
    final_rewards = agent.training_history['episode_rewards'][-100:]
    final_collisions = sum(agent.training_history['collisions'][-100:])
    final_avoidances = sum(agent.training_history['avoidances'][-100:])
    
    print(f"\n   Épisodes totaux: {num_episodes}")
    print(f"   Récompense moyenne (100 derniers): {sum(final_rewards)/len(final_rewards):.2f}")
    print(f"   Collisions (100 derniers): {final_collisions}")
    print(f"   Évitements (100 derniers): {final_avoidances}")
    print(f"   Epsilon final: {agent.epsilon:.4f}")
    
    # Statistiques de la table Q
    q_stats = agent.get_q_table_stats()
    print(f"\n   Table Q:")
    print(f"      Forme: {q_stats['shape']}")
    print(f"      Min/Max: {q_stats['min']:.2f} / {q_stats['max']:.2f}")
    print(f"      Moyenne: {q_stats['mean']:.2f}")
    print(f"      Valeurs non-nulles: {q_stats['non_zero']}")
    
    # Sauvegarder le modèle
    if save_model:
        agent.save(Q_TABLE_FILE)
    
    # Afficher la politique
    agent.print_policy()
    
    # Graphiques
    if show_plots:
        print("\n📈 Génération des graphiques...")
        
        plot_training_curves(
            agent.training_history,
            save_path=os.path.join(RESULTS_DIR, "training_curves.png"),
            show=True
        )
        
        plot_policy_visualization(
            agent.q_table,
            save_path=os.path.join(RESULTS_DIR, "policy.png"),
            show=True
        )
        
        plot_q_table_heatmap(
            agent.q_table,
            save_path=os.path.join(RESULTS_DIR, "q_table_heatmap.png"),
            show=True
        )
    
    print("\n✅ Entraînement terminé!")
    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement de l'agent Q-Learning"
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=NUM_EPISODES,
        help=f"Nombre d'épisodes (défaut: {NUM_EPISODES})"
    )
    parser.add_argument(
        '--max-steps', '-s',
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help=f"Max pas par épisode (défaut: {MAX_STEPS_PER_EPISODE})"
    )
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help="Afficher la simulation pendant l'entraînement"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Ne pas sauvegarder le modèle"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Ne pas afficher les graphiques"
    )
    
    args = parser.parse_args()
    
    train_q_learning(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_model=not args.no_save,
        show_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
