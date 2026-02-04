"""
Module utils - Utilitaires (génération de données, visualisation, métriques).
"""

from src.utils.dataset_generator import DatasetGenerator, generate_dataset_from_trained_agent
from src.utils.visualization import (
    plot_training_curves,
    plot_q_table_heatmap,
    plot_policy_visualization,
    plot_confusion_matrix,
    plot_comparison_results,
    plot_dataset_distribution
)
from src.utils.metrics import AgentEvaluator, analyze_stability, test_generalization

__all__ = [
    'DatasetGenerator',
    'generate_dataset_from_trained_agent',
    'plot_training_curves',
    'plot_q_table_heatmap',
    'plot_policy_visualization',
    'plot_confusion_matrix',
    'plot_comparison_results',
    'plot_dataset_distribution',
    'AgentEvaluator',
    'analyze_stability',
    'test_generalization'
]
