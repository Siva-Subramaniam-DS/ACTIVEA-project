"""(1+1) Evolution Strategy ((1+1)-ES) benchmark optimization module.

Implements the classical (1+1)-ES with Rechenberg's 1/5th success rule for
self-adaptive mutation step-size control, following Eiben & Smith (2015),
Chapter 6 ("Popular Evolutionary Algorithm Variants").
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import create_random_session, standard_mutation


class EvolutionStrategyOptimizer:
    """(1+1) Evolution Strategy optimizer with 1/5th success rule self-adaptation.
    
    A foundational EA variant maintaining a single parent that generates one offspring.
    Dynamically adjusts mutation intensity based on the recent empirical success rate.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        total_evaluations: int = 2500,
        initial_mutation_rate: float = 0.20,
        adaptation_window: int = 20,
        adaptation_factor: float = 0.85,
        checkpoint_interval: int = 50,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.total_evaluations = total_evaluations
        self.mutation_rate = initial_mutation_rate
        self.adaptation_window = adaptation_window
        self.adaptation_factor = adaptation_factor
        self.checkpoint_interval = checkpoint_interval
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs the (1+1)-ES optimization with self-adaptation."""
        start_time = time.time()
        self.history = []

        # Parent initialisation
        parent = create_random_session(self.activities_pool, rng=self.rng)
        parent_result = self.evaluator.evaluate(parent)
        parent_fitness = parent_result.total_fitness
        eval_count = 1

        self.best_individual = parent.copy()
        self.best_fitness = parent_fitness
        self.best_result = parent_result

        success_count = 0
        window_fitnesses: List[float] = [parent_fitness]

        while eval_count < self.total_evaluations:
            # Generate 1 offspring via mutation
            offspring = standard_mutation(
                parent,
                self.activities_pool,
                mutation_rate=self.mutation_rate,
                rng=self.rng,
            )
            offspring_result = self.evaluator.evaluate(offspring)
            offspring_fitness = offspring_result.total_fitness
            eval_count += 1
            window_fitnesses.append(offspring_fitness)

            # Plus-selection (1+1): accept if offspring is >= parent
            if offspring_fitness >= parent_fitness:
                parent = offspring
                parent_fitness = offspring_fitness
                parent_result = offspring_result
                success_count += 1

                if parent_fitness > self.best_fitness:
                    self.best_fitness = parent_fitness
                    self.best_individual = parent.copy()
                    self.best_result = parent_result

            # Rechenberg's 1/5th success rule adaptation check
            if eval_count % self.adaptation_window == 0:
                success_ratio = success_count / float(self.adaptation_window)
                if success_ratio > 0.20:
                    # Landscape is receptive: expand mutation scope
                    self.mutation_rate = min(0.50, self.mutation_rate / self.adaptation_factor)
                elif success_ratio < 0.20:
                    # Too many rejections: refine / focus search
                    self.mutation_rate = max(0.04, self.mutation_rate * self.adaptation_factor)
                success_count = 0

            # Checkpoint recording
            if eval_count % self.checkpoint_interval == 0 or eval_count == self.total_evaluations:
                mean_fit = sum(window_fitnesses) / len(window_fitnesses) if window_fitnesses else parent_fitness
                self.history.append({
                    "step": len(self.history),
                    "evaluation": eval_count,
                    "generation": eval_count // self.checkpoint_interval,
                    "best_fitness": round(self.best_fitness, 3),
                    "parent_fitness": round(parent_fitness, 3),
                    "mean_fitness": round(mean_fit, 3),
                    "mutation_rate": round(self.mutation_rate, 4),
                    "best_session": self.best_individual.to_tuple_list(),
                    "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                    "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
                })
                window_fitnesses = []

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
