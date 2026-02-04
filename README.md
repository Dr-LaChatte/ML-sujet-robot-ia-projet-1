# 🤖 Navette Robotique Anti-Collision

## Projet ML & RL - Entrepôt Logistique Automatisé

Ce projet implémente et compare deux approches d'apprentissage pour contrôler une navette AGV (Automated Guided Vehicle) dans un environnement d'entrepôt simulé :

- **Reinforcement Learning** : Q-learning tabulaire discret
- **Machine Learning Supervisé** : Classification k-NN

## 📋 Structure du Projet

```
ML-sujet-robot-ia-projet-1/
├── src/
│   ├── environment/          # Environnement de simulation Pygame
│   │   ├── warehouse_env.py  # Environnement principal
│   │   └── entities.py       # Navette et obstacles
│   ├── agents/
│   │   ├── q_learning_agent.py   # Agent Q-learning
│   │   └── knn_agent.py          # Agent k-NN supervisé
│   ├── utils/
│   │   ├── dataset_generator.py  # Génération du dataset
│   │   ├── visualization.py      # Visualisations
│   │   └── metrics.py            # Métriques de comparaison
│   └── config.py             # Configuration globale
├── scripts/
│   ├── train_rl.py           # Entraînement Q-learning
│   ├── generate_dataset.py   # Génération du dataset
│   ├── train_ml.py           # Entraînement k-NN
│   ├── compare_agents.py     # Comparaison des agents
│   └── demo.py               # Démonstration visuelle
├── data/                     # Datasets générés
├── models/                   # Modèles sauvegardés
├── results/                  # Résultats et graphiques
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Utilisation

### 1. Entraîner l'agent Q-learning (Phase 1)

```bash
python scripts/train_rl.py
```

### 2. Générer le dataset (Phase 2)

```bash
python scripts/generate_dataset.py
```

### 3. Entraîner le modèle k-NN (Phase 3)

```bash
python scripts/train_ml.py
```

### 4. Comparer les agents (Phase 4)

```bash
python scripts/compare_agents.py
```

### 5. Démonstration visuelle

```bash
python scripts/demo.py
```

## 🎮 Environnement

### États
- **Ligne de la navette** : Position verticale (0-4)
- **Distance obstacle** : Discrétisée (proche/moyen/loin)
- **Position Y obstacle** : Position verticale de l'obstacle le plus proche

### Actions
- `0` : Monter (déplacer vers le haut)
- `1` : Rester immobile
- `2` : Descendre (déplacer vers le bas)

### Récompenses
- `+1` : Évitement réussi (obstacle dépassé)
- `-100` : Collision
- `-0.1` : Pénalité par pas de temps

## 📊 Métriques de Comparaison

- Taux d'évitement
- Taux de collisions
- Stabilité temporelle
- Adaptation aux changements
- Généralisation

## 👥 Auteurs

Projet réalisé dans le cadre du M1 IA & Robotique.
