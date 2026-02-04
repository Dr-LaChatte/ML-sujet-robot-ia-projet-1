"""
Configuration globale du projet Navette Robotique Anti-Collision.
Contient tous les hyperparamètres et constantes du projet.
"""

# =============================================================================
# PARAMÈTRES DE L'ENVIRONNEMENT
# =============================================================================

# Dimensions de la fenêtre Pygame
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

# Grille de l'entrepôt
NUM_LANES = 5  # Nombre de lignes/couloirs
LANE_HEIGHT = WINDOW_HEIGHT // NUM_LANES

# Navette AGV
SHUTTLE_WIDTH = 60
SHUTTLE_HEIGHT = 50
SHUTTLE_START_X = 50  # Position X fixe de la navette
SHUTTLE_COLOR = (0, 120, 255)  # Bleu

# Obstacles (chariots élévateurs / humains)
OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 40
OBSTACLE_SPEED_MIN = 3
OBSTACLE_SPEED_MAX = 7
OBSTACLE_SPAWN_RATE = 60  # Frames entre chaque spawn (en moyenne)
OBSTACLE_COLOR = (255, 80, 80)  # Rouge

# =============================================================================
# DISCRÉTISATION DES ÉTATS
# =============================================================================

# Distance relative obstacle (en pixels)
DISTANCE_CLOSE = 150   # < 150 pixels = "proche"
DISTANCE_MEDIUM = 300  # 150-300 pixels = "moyen"
# > 300 pixels = "loin"

# Nombre d'états discrets pour la distance
NUM_DISTANCE_STATES = 3  # proche (0), moyen (1), loin (2)

# =============================================================================
# PARAMÈTRES Q-LEARNING
# =============================================================================

# Hyperparamètres
LEARNING_RATE = 0.1          # Alpha (α)
DISCOUNT_FACTOR = 0.95       # Gamma (γ)
EPSILON_START = 1.0          # Exploration initiale
EPSILON_END = 0.01           # Exploration minimale
EPSILON_DECAY = 0.9995       # Décroissance de epsilon

# Entraînement
NUM_EPISODES = 2000          # Nombre d'épisodes d'entraînement
MAX_STEPS_PER_EPISODE = 500  # Étapes max par épisode

# =============================================================================
# RÉCOMPENSES
# =============================================================================

REWARD_AVOID = 1.0           # Évitement réussi (obstacle dépassé)
REWARD_COLLISION = -100.0    # Collision
REWARD_STEP = -0.1           # Pénalité par pas de temps

# =============================================================================
# ACTIONS
# =============================================================================

ACTION_UP = 0       # Monter
ACTION_STAY = 1     # Rester
ACTION_DOWN = 2     # Descendre
NUM_ACTIONS = 3

ACTION_NAMES = {
    ACTION_UP: "Monter",
    ACTION_STAY: "Rester",
    ACTION_DOWN: "Descendre"
}

# =============================================================================
# PARAMÈTRES DATASET
# =============================================================================

DATASET_SIZE = 10000         # Nombre d'échantillons à générer
TEST_SIZE = 0.2              # 20% pour le test
VALIDATION_SIZE = 0.1        # 10% pour la validation

# =============================================================================
# PARAMÈTRES K-NN
# =============================================================================

KNN_NEIGHBORS = 5            # Nombre de voisins
KNN_WEIGHTS = 'distance'     # Pondération par distance

# =============================================================================
# PARAMÈTRES D'ÉVALUATION
# =============================================================================

EVAL_EPISODES = 100          # Épisodes pour l'évaluation comparative

# =============================================================================
# CHEMINS DES FICHIERS
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Fichiers
Q_TABLE_FILE = os.path.join(MODELS_DIR, "q_table.npy")
KNN_MODEL_FILE = os.path.join(MODELS_DIR, "knn_model.pkl")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")
TRAINING_HISTORY_FILE = os.path.join(RESULTS_DIR, "training_history.csv")

# =============================================================================
# COULEURS PYGAME
# =============================================================================

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (200, 200, 200)
COLOR_GREEN = (80, 200, 80)
COLOR_RED = (255, 80, 80)
COLOR_BLUE = (80, 80, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_FLOOR = (240, 230, 210)  # Couleur sol entrepôt
COLOR_LANE_MARKER = (180, 170, 150)  # Marquage au sol

# =============================================================================
# PARAMÈTRES D'AFFICHAGE
# =============================================================================

FPS = 60                     # Images par seconde
DISPLAY_TRAINING = False     # Afficher pendant l'entraînement RL
DISPLAY_GENERATION = False   # Afficher pendant la génération du dataset
