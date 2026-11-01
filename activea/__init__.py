"""ACTIVEA - Physical Activity Planning Evolutionary Algorithm Package
Coursework implementation for DG4COM Computational Intelligence (Aston University).
"""

from .models import Activity, ExerciseSession, FitnessResult
from .data_loader import ExerciseKnowledgeBase, WearableDataLoader
from .fitness import ActiveaFitnessEvaluator
from .operators import (
    tournament_selection,
    roulette_selection,
    uniform_crossover,
    single_point_crossover,
    standard_mutation,
    create_random_session,
    initialize_population,
)
from .core_ga import ActiveaGeneticAlgorithm
from .benchmark_sa import SimulatedAnnealingOptimizer
from .benchmark_random import RandomSearchOptimizer
from .benchmark_ils import IteratedLocalSearchOptimizer
from .benchmark_es import EvolutionStrategyOptimizer
from .benchmark_pso import ParticleSwarmOptimizer
from .benchmark_de import DifferentialEvolutionOptimizer
from .experiments import CrowdingGeneticAlgorithm, ParameterTuningExperiment
from .advanced import (
    BespokeRepairOperator,
    BespokeRepairGeneticAlgorithm,
    DomainPreferenceFitnessEvaluator,
)

__all__ = [
    "Activity",
    "ExerciseSession",
    "FitnessResult",
    "ExerciseKnowledgeBase",
    "WearableDataLoader",
    "ActiveaFitnessEvaluator",
    "tournament_selection",
    "roulette_selection",
    "uniform_crossover",
    "single_point_crossover",
    "standard_mutation",
    "create_random_session",
    "initialize_population",
    "ActiveaGeneticAlgorithm",
    "SimulatedAnnealingOptimizer",
    "RandomSearchOptimizer",
    "IteratedLocalSearchOptimizer",
    "EvolutionStrategyOptimizer",
    "ParticleSwarmOptimizer",
    "DifferentialEvolutionOptimizer",
    "CrowdingGeneticAlgorithm",
    "ParameterTuningExperiment",
    "BespokeRepairOperator",
    "BespokeRepairGeneticAlgorithm",
    "DomainPreferenceFitnessEvaluator",
]
