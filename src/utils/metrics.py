"""
Métriques de comparaison entre agents RL et ML.
"""

import numpy as np
from typing import Dict, List, Callable
from tqdm import tqdm
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import EVAL_EPISODES, MAX_STEPS_PER_EPISODE
from src.environment.warehouse_env import WarehouseEnvFast


class AgentEvaluator:
    """
    Évalue et compare les performances des agents.
    """
    
    def __init__(self):
        self.env = WarehouseEnvFast()
    
    def evaluate_agent(
        self,
        agent,
        agent_type: str,
        num_episodes: int = EVAL_EPISODES,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        verbose: bool = True
    ) -> Dict:
        """
        Évalue un agent sur plusieurs épisodes.
        
        Args:
            agent: Agent à évaluer (doit avoir une méthode choose_action)
            agent_type: 'rl' ou 'ml'
            num_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre max de pas par épisode
            verbose: Afficher la progression
            
        Returns:
            Dictionnaire de métriques
        """
        total_rewards = []
        episode_lengths = []
        collisions = 0
        avoidances = 0
        actions_taken = {0: 0, 1: 0, 2: 0}
        
        desc = f"Évaluation {agent_type.upper()}"
        
        for ep in tqdm(range(num_episodes), desc=desc, disable=not verbose):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                # Choisir l'action selon le type d'agent
                if agent_type == 'rl':
                    action = agent.choose_action(state, training=False)
                else:  # ml
                    components = self.env.get_state_components()
                    action = agent.choose_action_from_components(*components)
                
                actions_taken[action] += 1
                
                # Exécuter l'action
                next_state, reward, terminated, _, _ = self.env.step(action)
                episode_reward += reward
                
                if reward > 0:
                    avoidances += 1
                
                if terminated:
                    collisions += 1
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
        
        # Calculer les métriques
        total_obstacles = avoidances + collisions
        
        metrics = {
            'total_episodes': num_episodes,
            'total_score': sum(total_rewards),
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'collisions': collisions,
            'avoidances': avoidances,
            'collision_rate': 100 * collisions / total_obstacles if total_obstacles > 0 else 0,
            'avoidance_rate': 100 * avoidances / total_obstacles if total_obstacles > 0 else 0,
            'action_distribution': actions_taken,
            'rewards_list': total_rewards,
            'lengths_list': episode_lengths
        }
        
        return metrics
    
    def compare_agents(
        self,
        rl_agent,
        ml_agent,
        num_episodes: int = EVAL_EPISODES,
        verbose: bool = True
    ) -> Dict:
        """
        Compare deux agents (RL et ML).
        
        Args:
            rl_agent: Agent Q-learning
            ml_agent: Agent k-NN
            num_episodes: Nombre d'épisodes
            verbose: Afficher les résultats
            
        Returns:
            Dictionnaire avec les résultats de comparaison
        """
        print("\n" + "=" * 60)
        print("🔬 COMPARAISON DES AGENTS")
        print("=" * 60)
        
        # Évaluer les deux agents
        rl_metrics = self.evaluate_agent(rl_agent, 'rl', num_episodes)
        ml_metrics = self.evaluate_agent(ml_agent, 'ml', num_episodes)
        
        comparison = {
            'rl': rl_metrics,
            'ml': ml_metrics,
            'num_episodes': num_episodes
        }
        
        if verbose:
            self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comparison: Dict):
        """Affiche les résultats de comparaison."""
        rl = comparison['rl']
        ml = comparison['ml']
        
        print(f"\n📊 Résultats sur {comparison['num_episodes']} épisodes:")
        print("-" * 60)
        
        print(f"\n{'Métrique':<30s} {'RL (Q-Learning)':<20s} {'ML (k-NN)':<20s}")
        print("-" * 70)
        
        metrics_to_show = [
            ('Score total', 'total_score', '{:.0f}'),
            ('Récompense moyenne', 'avg_reward', '{:.2f}'),
            ('Écart-type récompense', 'std_reward', '{:.2f}'),
            ('Durée moyenne épisode', 'avg_episode_length', '{:.1f}'),
            ('Collisions', 'collisions', '{:.0f}'),
            ('Évitements', 'avoidances', '{:.0f}'),
            ('Taux collision (%)', 'collision_rate', '{:.1f}'),
            ('Taux évitement (%)', 'avoidance_rate', '{:.1f}'),
        ]
        
        for name, key, fmt in metrics_to_show:
            rl_val = fmt.format(rl[key])
            ml_val = fmt.format(ml[key])
            print(f"{name:<30s} {rl_val:<20s} {ml_val:<20s}")
        
        # Déterminer le gagnant
        print("\n" + "=" * 60)
        print("🏆 ANALYSE:")
        
        if rl['avoidance_rate'] > ml['avoidance_rate']:
            print(f"   ✅ RL a un meilleur taux d'évitement "
                  f"({rl['avoidance_rate']:.1f}% vs {ml['avoidance_rate']:.1f}%)")
        elif ml['avoidance_rate'] > rl['avoidance_rate']:
            print(f"   ✅ ML a un meilleur taux d'évitement "
                  f"({ml['avoidance_rate']:.1f}% vs {rl['avoidance_rate']:.1f}%)")
        else:
            print(f"   🤝 Taux d'évitement identiques ({rl['avoidance_rate']:.1f}%)")
        
        if rl['avg_episode_length'] > ml['avg_episode_length']:
            print(f"   ✅ RL survit plus longtemps en moyenne "
                  f"({rl['avg_episode_length']:.0f} vs {ml['avg_episode_length']:.0f} pas)")
        elif ml['avg_episode_length'] > rl['avg_episode_length']:
            print(f"   ✅ ML survit plus longtemps en moyenne "
                  f"({ml['avg_episode_length']:.0f} vs {rl['avg_episode_length']:.0f} pas)")


def analyze_stability(
    agent,
    agent_type: str,
    num_runs: int = 5,
    episodes_per_run: int = 50
) -> Dict:
    """
    Analyse la stabilité d'un agent sur plusieurs runs.
    
    Args:
        agent: Agent à analyser
        agent_type: 'rl' ou 'ml'
        num_runs: Nombre de runs
        episodes_per_run: Épisodes par run
        
    Returns:
        Métriques de stabilité
    """
    evaluator = AgentEvaluator()
    
    run_scores = []
    run_avoidance_rates = []
    
    print(f"\n📈 Analyse de stabilité ({agent_type.upper()})...")
    
    for run in range(num_runs):
        metrics = evaluator.evaluate_agent(
            agent, agent_type, episodes_per_run, verbose=False
        )
        run_scores.append(metrics['avg_reward'])
        run_avoidance_rates.append(metrics['avoidance_rate'])
    
    stability = {
        'mean_score': np.mean(run_scores),
        'std_score': np.std(run_scores),
        'cv_score': np.std(run_scores) / abs(np.mean(run_scores)) if np.mean(run_scores) != 0 else 0,
        'mean_avoidance': np.mean(run_avoidance_rates),
        'std_avoidance': np.std(run_avoidance_rates),
        'scores': run_scores,
        'avoidance_rates': run_avoidance_rates
    }
    
    print(f"   Score moyen: {stability['mean_score']:.2f} (+/- {stability['std_score']:.2f})")
    print(f"   Coefficient de variation: {stability['cv_score']:.2%}")
    print(f"   Taux évitement: {stability['mean_avoidance']:.1f}% (+/- {stability['std_avoidance']:.1f}%)")
    
    return stability


def test_generalization(
    agent,
    agent_type: str,
    scenarios: List[Dict],
    episodes_per_scenario: int = 30
) -> Dict:
    """
    Teste la généralisation d'un agent sur différents scénarios.
    
    Args:
        agent: Agent à tester
        agent_type: 'rl' ou 'ml'
        scenarios: Liste de configurations différentes
        episodes_per_scenario: Épisodes par scénario
        
    Returns:
        Résultats par scénario
    """
    evaluator = AgentEvaluator()
    results = {}
    
    print(f"\n🧪 Test de généralisation ({agent_type.upper()})...")
    
    for scenario in scenarios:
        name = scenario.get('name', 'default')
        print(f"\n   Scénario: {name}")
        
        # TODO: Modifier l'environnement selon le scénario
        # (vitesse obstacles, fréquence spawn, etc.)
        
        metrics = evaluator.evaluate_agent(
            agent, agent_type, episodes_per_scenario, verbose=False
        )
        
        results[name] = {
            'avoidance_rate': metrics['avoidance_rate'],
            'avg_reward': metrics['avg_reward'],
            'avg_length': metrics['avg_episode_length']
        }
        
        print(f"      Taux évitement: {metrics['avoidance_rate']:.1f}%")
    
    return results
