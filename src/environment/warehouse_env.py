"""
Environnement de simulation de l'entrepôt logistique.
Compatible avec l'interface Gymnasium pour le Q-learning.
"""

import pygame
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, NUM_LANES, LANE_HEIGHT,
    OBSTACLE_SPAWN_RATE, DISTANCE_CLOSE, DISTANCE_MEDIUM,
    NUM_DISTANCE_STATES, NUM_ACTIONS,
    ACTION_UP, ACTION_STAY, ACTION_DOWN,
    REWARD_AVOID, REWARD_COLLISION, REWARD_STEP,
    COLOR_FLOOR, COLOR_LANE_MARKER, COLOR_WHITE, COLOR_BLACK,
    FPS, SHUTTLE_START_X, SHUTTLE_WIDTH
)
from src.environment.entities import Shuttle, Obstacle


class WarehouseEnv:
    """
    Environnement de simulation d'entrepôt logistique.
    
    Une navette AGV doit éviter des obstacles mobiles (chariots/humains)
    qui traversent son chemin horizontalement.
    
    États (discrétisés):
        - Ligne de la navette (0 à NUM_LANES-1)
        - Distance de l'obstacle le plus proche (proche/moyen/loin)
        - Ligne de l'obstacle le plus proche (0 à NUM_LANES-1)
    
    Actions:
        - 0: Monter
        - 1: Rester
        - 2: Descendre
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialise l'environnement.
        
        Args:
            render_mode: 'human' pour affichage Pygame, None sinon
        """
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        # Entités
        self.shuttle = Shuttle()
        self.obstacles = []
        
        # Compteurs
        self.steps = 0
        self.total_avoided = 0
        self.total_collisions = 0
        self.spawn_timer = 0
        
        # Espace d'états et d'actions
        self.num_states = NUM_LANES * NUM_DISTANCE_STATES * NUM_LANES
        self.num_actions = NUM_ACTIONS
        
        # Initialisation Pygame si nécessaire
        if self.render_mode == 'human':
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialise Pygame pour l'affichage."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("🏭 Entrepôt Logistique - Navette AGV")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Réinitialise l'environnement.
        
        Args:
            seed: Graine aléatoire (optionnel)
            
        Returns:
            (état_initial, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset des entités
        self.shuttle.reset(lane=NUM_LANES // 2)  # Commence au milieu
        self.obstacles = []
        
        # Reset des compteurs
        self.steps = 0
        self.spawn_timer = 0
        
        # Spawn initial d'un obstacle
        self._spawn_obstacle()
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: Action à effectuer (0=Monter, 1=Rester, 2=Descendre)
            
        Returns:
            (nouvel_état, récompense, terminé, tronqué, info)
        """
        self.steps += 1
        reward = REWARD_STEP  # Pénalité par défaut
        terminated = False
        truncated = False
        
        # Exécuter l'action
        if action == ACTION_UP:
            self.shuttle.move_up()
        elif action == ACTION_DOWN:
            self.shuttle.move_down()
        # ACTION_STAY ne fait rien
        
        # Mettre à jour les obstacles
        for obstacle in self.obstacles:
            obstacle.update()
            
            # Vérifier si l'obstacle a dépassé la navette
            if obstacle.has_passed_shuttle(SHUTTLE_START_X):
                reward += REWARD_AVOID
                self.total_avoided += 1
        
        # Supprimer les obstacles hors écran
        self.obstacles = [obs for obs in self.obstacles if not obs.is_off_screen()]
        
        # Spawn de nouveaux obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= OBSTACLE_SPAWN_RATE:
            if random.random() < 0.7:  # 70% de chance de spawn
                self._spawn_obstacle()
            self.spawn_timer = 0
        
        # Vérifier les collisions
        shuttle_rect = self.shuttle.get_rect()
        for obstacle in self.obstacles:
            if shuttle_rect.colliderect(obstacle.get_rect()):
                reward = REWARD_COLLISION
                terminated = True
                self.total_collisions += 1
                break
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def _spawn_obstacle(self):
        """Crée un nouvel obstacle."""
        # Éviter de spawn sur la même ligne qu'un obstacle proche
        available_lanes = list(range(NUM_LANES))
        for obs in self.obstacles:
            if obs.x > WINDOW_WIDTH * 0.7:  # Obstacle récent
                if obs.lane in available_lanes:
                    available_lanes.remove(obs.lane)
        
        if available_lanes:
            lane = random.choice(available_lanes)
        else:
            lane = random.randint(0, NUM_LANES - 1)
        
        self.obstacles.append(Obstacle(lane=lane))
    
    def _get_state(self) -> int:
        """
        Calcule l'état discret actuel.
        
        Returns:
            État sous forme d'entier unique
        """
        # Position de la navette
        shuttle_lane = self.shuttle.lane
        
        # Trouver l'obstacle le plus proche devant la navette
        closest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            distance = obstacle.get_distance_to(SHUTTLE_START_X + SHUTTLE_WIDTH)
            if 0 < distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle
        
        # Discrétiser la distance
        if closest_obstacle is None:
            distance_state = 2  # Loin (pas d'obstacle)
            obstacle_lane = shuttle_lane  # Par défaut, même ligne
        else:
            if min_distance < DISTANCE_CLOSE:
                distance_state = 0  # Proche
            elif min_distance < DISTANCE_MEDIUM:
                distance_state = 1  # Moyen
            else:
                distance_state = 2  # Loin
            obstacle_lane = closest_obstacle.lane
        
        # Encoder l'état en un seul entier
        state = (shuttle_lane * NUM_DISTANCE_STATES * NUM_LANES +
                 distance_state * NUM_LANES +
                 obstacle_lane)
        
        return state
    
    def get_state_components(self) -> Tuple[int, int, int]:
        """
        Retourne les composantes de l'état actuel (pour le ML supervisé).
        
        Returns:
            (ligne_navette, distance_state, ligne_obstacle)
        """
        shuttle_lane = self.shuttle.lane
        
        closest_obstacle = None
        min_distance = float('inf')
        
        for obstacle in self.obstacles:
            distance = obstacle.get_distance_to(SHUTTLE_START_X + SHUTTLE_WIDTH)
            if 0 < distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle
        
        if closest_obstacle is None:
            distance_state = 2
            obstacle_lane = shuttle_lane
        else:
            if min_distance < DISTANCE_CLOSE:
                distance_state = 0
            elif min_distance < DISTANCE_MEDIUM:
                distance_state = 1
            else:
                distance_state = 2
            obstacle_lane = closest_obstacle.lane
        
        return shuttle_lane, distance_state, obstacle_lane
    
    def _get_info(self) -> Dict[str, Any]:
        """Retourne des informations supplémentaires."""
        return {
            'steps': self.steps,
            'total_avoided': self.total_avoided,
            'total_collisions': self.total_collisions,
            'num_obstacles': len(self.obstacles)
        }
    
    def render(self):
        """Affiche l'environnement."""
        if self.render_mode != 'human':
            return
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        
        # Effacer l'écran
        self.screen.fill(COLOR_FLOOR)
        
        # Dessiner les lignes de couloir
        for i in range(1, NUM_LANES):
            y = i * LANE_HEIGHT
            pygame.draw.line(self.screen, COLOR_LANE_MARKER, (0, y), (WINDOW_WIDTH, y), 2)
        
        # Dessiner les marquages au sol (effet de mouvement)
        for i in range(NUM_LANES):
            y = i * LANE_HEIGHT + LANE_HEIGHT // 2
            for x in range(0, WINDOW_WIDTH, 100):
                offset = (self.steps * 3) % 100
                pygame.draw.rect(self.screen, COLOR_LANE_MARKER, 
                               (x - offset, y - 2, 40, 4))
        
        # Dessiner les obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        # Dessiner la navette
        self.shuttle.draw(self.screen)
        
        # Afficher les informations
        info_texts = [
            f"Étapes: {self.steps}",
            f"Évités: {self.total_avoided}",
            f"Collisions: {self.total_collisions}",
            f"Ligne: {self.shuttle.lane + 1}/{NUM_LANES}"
        ]
        
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, COLOR_BLACK)
            self.screen.blit(surface, (WINDOW_WIDTH - 150, 10 + i * 25))
        
        # Afficher l'état actuel
        shuttle_lane, dist_state, obs_lane = self.get_state_components()
        dist_names = ["Proche", "Moyen", "Loin"]
        state_text = f"État: L{shuttle_lane+1}-{dist_names[dist_state]}-O{obs_lane+1}"
        surface = self.font.render(state_text, True, COLOR_BLACK)
        self.screen.blit(surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        
        return True
    
    def close(self):
        """Ferme l'environnement."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


class WarehouseEnvFast:
    """
    Version optimisée de l'environnement sans Pygame.
    Pour l'entraînement rapide.
    """
    
    def __init__(self):
        self.shuttle_lane = NUM_LANES // 2
        self.obstacles = []  # Liste de tuples (lane, x, speed)
        self.steps = 0
        self.total_avoided = 0
        self.total_collisions = 0
        self.spawn_timer = 0
        
        self.num_states = NUM_LANES * NUM_DISTANCE_STATES * NUM_LANES
        self.num_actions = NUM_ACTIONS
    
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        
        self.shuttle_lane = NUM_LANES // 2
        self.obstacles = []
        self.steps = 0
        self.spawn_timer = 0
        
        self._spawn_obstacle()
        
        return self._get_state(), {}
    
    def step(self, action):
        self.steps += 1
        reward = REWARD_STEP
        terminated = False
        
        # Action
        if action == ACTION_UP and self.shuttle_lane > 0:
            self.shuttle_lane -= 1
        elif action == ACTION_DOWN and self.shuttle_lane < NUM_LANES - 1:
            self.shuttle_lane += 1
        
        # Update obstacles
        new_obstacles = []
        for lane, x, speed in self.obstacles:
            new_x = x - speed
            
            # Check if passed
            if x > SHUTTLE_START_X + SHUTTLE_WIDTH and new_x <= SHUTTLE_START_X + SHUTTLE_WIDTH:
                reward += REWARD_AVOID
                self.total_avoided += 1
            
            # Check collision
            if (lane == self.shuttle_lane and 
                new_x < SHUTTLE_START_X + SHUTTLE_WIDTH and 
                new_x + 50 > SHUTTLE_START_X):  # 50 = obstacle width
                reward = REWARD_COLLISION
                terminated = True
                self.total_collisions += 1
            
            # Keep if on screen
            if new_x + 50 > 0:
                new_obstacles.append((lane, new_x, speed))
        
        self.obstacles = new_obstacles
        
        # Spawn
        self.spawn_timer += 1
        if self.spawn_timer >= OBSTACLE_SPAWN_RATE:
            if random.random() < 0.7:
                self._spawn_obstacle()
            self.spawn_timer = 0
        
        return self._get_state(), reward, terminated, False, {}
    
    def _spawn_obstacle(self):
        lane = random.randint(0, NUM_LANES - 1)
        speed = random.uniform(3, 7)
        self.obstacles.append((lane, WINDOW_WIDTH + 50, speed))
    
    def _get_state(self):
        shuttle_lane = self.shuttle_lane
        
        # Find closest
        closest_dist = float('inf')
        closest_lane = shuttle_lane
        
        for lane, x, _ in self.obstacles:
            dist = x - (SHUTTLE_START_X + SHUTTLE_WIDTH)
            if 0 < dist < closest_dist:
                closest_dist = dist
                closest_lane = lane
        
        # Discretize
        if closest_dist == float('inf'):
            distance_state = 2
        elif closest_dist < DISTANCE_CLOSE:
            distance_state = 0
        elif closest_dist < DISTANCE_MEDIUM:
            distance_state = 1
        else:
            distance_state = 2
        
        return (shuttle_lane * NUM_DISTANCE_STATES * NUM_LANES +
                distance_state * NUM_LANES +
                closest_lane)
    
    def get_state_components(self):
        shuttle_lane = self.shuttle_lane
        
        closest_dist = float('inf')
        closest_lane = shuttle_lane
        
        for lane, x, _ in self.obstacles:
            dist = x - (SHUTTLE_START_X + SHUTTLE_WIDTH)
            if 0 < dist < closest_dist:
                closest_dist = dist
                closest_lane = lane
        
        if closest_dist == float('inf'):
            distance_state = 2
        elif closest_dist < DISTANCE_CLOSE:
            distance_state = 0
        elif closest_dist < DISTANCE_MEDIUM:
            distance_state = 1
        else:
            distance_state = 2
        
        return shuttle_lane, distance_state, closest_lane
