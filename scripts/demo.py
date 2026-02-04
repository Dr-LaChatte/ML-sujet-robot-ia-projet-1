"""
Script de démonstration visuelle.
Lance la simulation avec un agent (RL ou ML) pour visualiser le comportement.
"""

import sys
import os
import argparse
import time

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Q_TABLE_FILE, KNN_MODEL_FILE, MAX_STEPS_PER_EPISODE, FPS
from src.environment.warehouse_env import WarehouseEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.knn_agent import KNNAgent


def run_demo(
    agent_type: str = 'rl',
    agent_path: str = None,
    num_episodes: int = 5,
    max_steps: int = MAX_STEPS_PER_EPISODE,
    speed: float = 1.0
):
    """
    Lance une démonstration visuelle.
    
    Args:
        agent_type: 'rl' ou 'ml'
        agent_path: Chemin de l'agent
        num_episodes: Nombre d'épisodes à jouer
        max_steps: Nombre max de pas par épisode
        speed: Vitesse de simulation (1.0 = normal)
    """
    print("=" * 60)
    print("🎮 DÉMONSTRATION VISUELLE")
    print("=" * 60)
    
    # Charger l'agent
    print(f"\n📥 Chargement de l'agent {agent_type.upper()}...")
    
    if agent_type == 'rl':
        agent = QLearningAgent()
        if agent_path is None:
            agent_path = Q_TABLE_FILE
        try:
            agent.load(agent_path)
        except FileNotFoundError:
            print(f"\n❌ Erreur: Agent RL non trouvé à {agent_path}")
            print("   Veuillez d'abord exécuter train_rl.py")
            return
    else:
        agent = KNNAgent()
        if agent_path is None:
            agent_path = KNN_MODEL_FILE
        try:
            agent.load(agent_path)
        except FileNotFoundError:
            print(f"\n❌ Erreur: Agent ML non trouvé à {agent_path}")
            print("   Veuillez d'abord exécuter train_ml.py")
            return
    
    print(f"   ✅ Agent chargé: {agent_path}")
    
    # Créer l'environnement avec affichage
    env = WarehouseEnv(render_mode='human')
    
    print(f"\n📋 Configuration:")
    print(f"   Agent: {agent_type.upper()}")
    print(f"   Épisodes: {num_episodes}")
    print(f"   Max pas/épisode: {max_steps}")
    print(f"   Vitesse: {speed}x")
    
    print(f"\n🚀 Lancement de la démonstration...")
    print("   (Fermez la fenêtre pour quitter)")
    
    total_rewards = []
    total_avoidances = 0
    total_collisions = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        avoidances = 0
        
        print(f"\n   📍 Épisode {episode + 1}/{num_episodes}")
        
        for step in range(max_steps):
            # Choisir l'action
            if agent_type == 'rl':
                action = agent.choose_action(state, training=False)
            else:
                components = env.get_state_components()
                action = agent.choose_action_from_components(*components)
            
            # Exécuter l'action
            next_state, reward, terminated, _, info = env.step(action)
            episode_reward += reward
            
            if reward > 0:
                avoidances += 1
                total_avoidances += 1
            
            # Afficher
            continue_rendering = env.render()
            
            if not continue_rendering:
                print("\n   ❌ Fenêtre fermée.")
                env.close()
                return
            
            # Ajuster la vitesse
            if speed < 1.0:
                time.sleep((1.0 - speed) * 0.02)
            
            if terminated:
                total_collisions += 1
                print(f"      💥 Collision après {step + 1} pas (évitements: {avoidances})")
                break
            
            state = next_state
        
        if not terminated:
            print(f"      ✅ Survécu {max_steps} pas (évitements: {avoidances})")
        
        total_rewards.append(episode_reward)
        
        # Pause entre les épisodes
        time.sleep(0.5)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE LA DÉMONSTRATION")
    print("=" * 60)
    print(f"\n   Agent: {agent_type.upper()}")
    print(f"   Épisodes: {num_episodes}")
    print(f"   Récompense totale: {sum(total_rewards):.0f}")
    print(f"   Récompense moyenne: {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"   Évitements: {total_avoidances}")
    print(f"   Collisions: {total_collisions}")
    print(f"   Taux de survie: {100*(num_episodes-total_collisions)/num_episodes:.1f}%")
    
    print("\n   Fermez la fenêtre pour quitter...")
    
    # Garder la fenêtre ouverte
    import pygame
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    env.close()
    print("\n✅ Démonstration terminée!")


def run_manual_mode():
    """
    Mode manuel : contrôle de la navette au clavier.
    """
    print("=" * 60)
    print("🎮 MODE MANUEL")
    print("=" * 60)
    print("\n   Contrôles:")
    print("   ↑ (Flèche haut) : Monter")
    print("   ↓ (Flèche bas)  : Descendre")
    print("   Espace          : Rester")
    print("   R               : Recommencer")
    print("   Q / Échap       : Quitter")
    
    import pygame
    
    env = WarehouseEnv(render_mode='human')
    state, _ = env.reset()
    
    running = True
    total_reward = 0
    avoidances = 0
    
    while running:
        action = 1  # Rester par défaut
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Monter
                elif event.key == pygame.K_DOWN:
                    action = 2  # Descendre
                elif event.key == pygame.K_SPACE:
                    action = 1  # Rester
                elif event.key == pygame.K_r:
                    state, _ = env.reset()
                    total_reward = 0
                    avoidances = 0
                    print("\n   🔄 Nouvelle partie!")
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
        
        if running:
            state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            
            if reward > 0:
                avoidances += 1
            
            env.render()
            
            if terminated:
                print(f"\n   💥 Collision! Score: {total_reward:.0f}, Évitements: {avoidances}")
                print("   Appuyez sur R pour recommencer ou Q pour quitter.")
    
    env.close()
    print("\n✅ Mode manuel terminé!")


def main():
    parser = argparse.ArgumentParser(
        description="Démonstration visuelle de la navette robotique"
    )
    parser.add_argument(
        '--agent', '-a',
        type=str,
        choices=['rl', 'ml', 'manual'],
        default='rl',
        help="Type d'agent (rl, ml, ou manual pour jouer)"
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help="Chemin de l'agent"
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=5,
        help="Nombre d'épisodes (défaut: 5)"
    )
    parser.add_argument(
        '--max-steps', '-s',
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help=f"Max pas par épisode (défaut: {MAX_STEPS_PER_EPISODE})"
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help="Vitesse de simulation (défaut: 1.0)"
    )
    
    args = parser.parse_args()
    
    if args.agent == 'manual':
        run_manual_mode()
    else:
        run_demo(
            agent_type=args.agent,
            agent_path=args.path,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            speed=args.speed
        )


if __name__ == "__main__":
    main()
