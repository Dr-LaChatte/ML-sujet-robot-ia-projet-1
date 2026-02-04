"""
Entités du jeu : Navette AGV et Obstacles.
"""

import pygame
import random
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (
    SHUTTLE_WIDTH, SHUTTLE_HEIGHT, SHUTTLE_COLOR, SHUTTLE_START_X,
    OBSTACLE_WIDTH, OBSTACLE_HEIGHT, OBSTACLE_COLOR,
    OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX,
    NUM_LANES, LANE_HEIGHT, WINDOW_WIDTH
)


class Shuttle:
    """
    Navette AGV (Automated Guided Vehicle).
    Se déplace uniquement verticalement entre les lignes.
    """
    
    def __init__(self, start_lane: int = 2):
        """
        Initialise la navette.
        
        Args:
            start_lane: Ligne de départ (0 à NUM_LANES-1)
        """
        self.lane = start_lane
        self.x = SHUTTLE_START_X
        self.width = SHUTTLE_WIDTH
        self.height = SHUTTLE_HEIGHT
        self.color = SHUTTLE_COLOR
        self._update_y()
    
    def _update_y(self):
        """Met à jour la position Y en fonction de la ligne."""
        # Centre la navette dans sa ligne
        self.y = self.lane * LANE_HEIGHT + (LANE_HEIGHT - self.height) // 2
    
    def move_up(self) -> bool:
        """
        Déplace la navette vers le haut (ligne inférieure en numéro).
        
        Returns:
            True si le mouvement a été effectué, False sinon
        """
        if self.lane > 0:
            self.lane -= 1
            self._update_y()
            return True
        return False
    
    def move_down(self) -> bool:
        """
        Déplace la navette vers le bas (ligne supérieure en numéro).
        
        Returns:
            True si le mouvement a été effectué, False sinon
        """
        if self.lane < NUM_LANES - 1:
            self.lane += 1
            self._update_y()
            return True
        return False
    
    def stay(self):
        """La navette reste immobile."""
        pass
    
    def get_rect(self) -> pygame.Rect:
        """Retourne le rectangle de collision."""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def reset(self, lane: int = None):
        """
        Réinitialise la navette.
        
        Args:
            lane: Nouvelle ligne (ou aléatoire si None)
        """
        if lane is None:
            self.lane = random.randint(0, NUM_LANES - 1)
        else:
            self.lane = lane
        self._update_y()
    
    def draw(self, screen: pygame.Surface):
        """
        Dessine la navette sur l'écran.
        
        Args:
            screen: Surface Pygame
        """
        # Corps principal
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Bordure
        pygame.draw.rect(screen, (0, 80, 180), (self.x, self.y, self.width, self.height), 3)
        
        # Roues
        wheel_color = (50, 50, 50)
        wheel_radius = 6
        pygame.draw.circle(screen, wheel_color, (self.x + 10, self.y + self.height), wheel_radius)
        pygame.draw.circle(screen, wheel_color, (self.x + self.width - 10, self.y + self.height), wheel_radius)
        pygame.draw.circle(screen, wheel_color, (self.x + 10, self.y), wheel_radius)
        pygame.draw.circle(screen, wheel_color, (self.x + self.width - 10, self.y), wheel_radius)
        
        # Capteur avant (représentation visuelle)
        sensor_color = (255, 255, 0)
        pygame.draw.polygon(screen, sensor_color, [
            (self.x + self.width, self.y + self.height // 2 - 5),
            (self.x + self.width + 15, self.y + self.height // 2),
            (self.x + self.width, self.y + self.height // 2 + 5)
        ])


class Obstacle:
    """
    Obstacle mobile (chariot élévateur ou humain).
    Se déplace horizontalement de droite à gauche.
    """
    
    def __init__(self, lane: int = None, speed: float = None):
        """
        Initialise l'obstacle.
        
        Args:
            lane: Ligne de l'obstacle (aléatoire si None)
            speed: Vitesse de déplacement (aléatoire si None)
        """
        self.lane = lane if lane is not None else random.randint(0, NUM_LANES - 1)
        self.speed = speed if speed is not None else random.uniform(OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX)
        self.x = WINDOW_WIDTH + OBSTACLE_WIDTH  # Commence hors écran à droite
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
        self.color = OBSTACLE_COLOR
        self.passed = False  # True si l'obstacle a dépassé la navette
        self._update_y()
    
    def _update_y(self):
        """Met à jour la position Y en fonction de la ligne."""
        self.y = self.lane * LANE_HEIGHT + (LANE_HEIGHT - self.height) // 2
    
    def update(self):
        """Met à jour la position de l'obstacle."""
        self.x -= self.speed
    
    def is_off_screen(self) -> bool:
        """Vérifie si l'obstacle est sorti de l'écran."""
        return self.x + self.width < 0
    
    def has_passed_shuttle(self, shuttle_x: int) -> bool:
        """
        Vérifie si l'obstacle a dépassé la navette.
        
        Args:
            shuttle_x: Position X de la navette
            
        Returns:
            True si l'obstacle vient de dépasser la navette
        """
        if not self.passed and self.x + self.width < shuttle_x:
            self.passed = True
            return True
        return False
    
    def get_rect(self) -> pygame.Rect:
        """Retourne le rectangle de collision."""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def get_distance_to(self, shuttle_x: int) -> float:
        """
        Calcule la distance horizontale à la navette.
        
        Args:
            shuttle_x: Position X de la navette
            
        Returns:
            Distance en pixels (peut être négative si dépassé)
        """
        return self.x - shuttle_x
    
    def draw(self, screen: pygame.Surface):
        """
        Dessine l'obstacle sur l'écran.
        
        Args:
            screen: Surface Pygame
        """
        # Corps principal
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Bordure
        pygame.draw.rect(screen, (180, 50, 50), (self.x, self.y, self.width, self.height), 2)
        
        # Symbole d'avertissement
        warning_color = (255, 255, 100)
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        pygame.draw.polygon(screen, warning_color, [
            (center_x, center_y - 12),
            (center_x - 10, center_y + 8),
            (center_x + 10, center_y + 8)
        ])
        pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y + 3), 2)
        pygame.draw.line(screen, (0, 0, 0), (center_x, center_y - 6), (center_x, center_y - 1), 2)
