"""Iterated Local Search (ILS) benchmark optimization module for ACTIVEA.

Implements Iterated Local Search based on the Computational Intelligence lecture
slides on Local Search: combining perturbative hill-climbing search with periodic
stochastic perturbation ("kicks") and random restarts to escape local optima.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import create_random_session, standard_mutation


class IteratedLocalSearchOptimizer:
    """Iterated Local Search optimizer for exercise session design.
    
    Combines greedy hill-climbing exploitation with escaping mechanisms
    (stochastic perturbation kicks and restarts on stagnation).
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        total_evaluations: int = 2500,
        stagnation_limit: int = 30,
        checkpoint_interval: int = 50,
        mutation_rate: float = 0.20,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.total_evaluations = total_evaluations
        self.stagnation_limit = stagnation_limit
        self.checkpoint_interval = checkpoint_interval
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs the Iterated Local Search optimization."""
        start_time = time.time()
        self.history = []

        # Initial solution
        current_solution = create_random_session(self.activities_pool, rng=self.rng)
        current_result = self.evaluator.evaluate(current_solution)
        current_fitness = current_result.total_fitness
        eval_count = 1

        self.best_individual = current_solution.copy()
        self.best_fitness = current_fitness
        self.best_result = current_result

        stagnation_count = 0
        window_fitnesses: List[float] = [current_fitness]

        while eval_count < self.total_evaluations:
            # Check for stagnation / kick or restart
            if stagnation_count >= self.stagnation_limit:
                if self.rng.random() < 0.4:
                    # Random restart
                    current_solution = create_random_session(self.activities_pool, rng=self.rng)
                else:
                    # Large perturbation (multi-step kick)
                    mutated = self.best_individual.copy()
                    for _ in range(3):
                        mutated = standard_mutation(
                            mutated, self.activities_pool, mutation_rate=0.4, rng=self.rng
                        )
                    current_solution = mutated
                
                current_result = self.evaluator.evaluate(current_solution)
                current_fitness = current_result.total_fitness
                eval_count += 1
                stagnation_count = 0
                window_fitnesses.append(current_fitness)

                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_individual = current_solution.copy()
                    self.best_result = current_result

                if eval_count >= self.total_evaluations:
                    break

            # Local search step: perturb current solution
            neighbor = standard_mutation(
                current_solution,
                self.activities_pool,
                mutation_rate=self.mutation_rate,
                rng=self.rng,
            )
            neighbor_result = self.evaluator.evaluate(neighbor)
            neighbor_fitness = neighbor_result.total_fitness
            eval_count += 1
            window_fitnesses.append(neighbor_fitness)

            # Greedy acceptance
            if neighbor_fitness >= current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                current_result = neighbor_result
                stagnation_count = 0

                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_individual = current_solution.copy()
                    self.best_result = current_result
            else:
                stagnation_count += 1

            # Checkpoint recording
            if eval_count % self.checkpoint_interval == 0 or eval_count == self.total_evaluations:
                mean_fit = sum(window_fitnesses) / len(window_fitnesses) if window_fitnesses else current_fitness
                self.history.append({
                    "step": len(self.history),
                    "evaluation": eval_count,
                    "generation": eval_count // self.checkpoint_interval,
                    "best_fitness": round(self.best_fitness, 3),
                    "current_fitness": round(current_fitness, 3),
                    "mean_fitness": round(mean_fit, 3),
                    "best_session": self.best_individual.to_tuple_list(),
                    "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                    "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
                })
                window_fitnesses = []

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
