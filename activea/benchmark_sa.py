"""Nature-inspired optimization benchmark module for Task 4.

Implements Simulated Annealing (SA), a foundational physics/nature-inspired
stochastic trajectory optimization method, to provide a direct comparative
benchmark against ACTIVEA's population-based evolutionary search.
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import create_random_session, standard_mutation


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer operating on exercise session chromosomes.
    
    Provides an apples-to-apples comparison using the exact same fitness evaluator
    and representation under equivalent computational budgets.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        total_evaluations: int = 3000,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.01,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.total_evaluations = total_evaluations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs the Simulated Annealing optimization process."""
        start_time = time.time()
        self.history = []

        # Initial solution
        current_solution = create_random_session(self.activities_pool, rng=self.rng)
        current_result = self.evaluator.evaluate(current_solution)
        current_fitness = current_result.total_fitness

        self.best_individual = current_solution.copy()
        self.best_fitness = current_fitness
        self.best_result = current_result

        temperature = self.initial_temperature
        eval_count = 1

        # Checkpoint recording step (to align with GA generations)
        step_interval = max(1, self.total_evaluations // 60)

        self.history.append({
            "step": 0,
            "evaluation": 1,
            "temperature": round(temperature, 3),
            "current_fitness": round(current_fitness, 3),
            "best_fitness": round(self.best_fitness, 3),
            "best_session": self.best_individual.to_tuple_list(),
            "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
            "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
        })

        while eval_count < self.total_evaluations and temperature > self.min_temperature:
            # Generate neighbor via perturbation/mutation
            neighbor = standard_mutation(
                current_solution,
                self.activities_pool,
                mutation_rate=0.25,
                rng=self.rng
            )
            neighbor_result = self.evaluator.evaluate(neighbor)
            neighbor_fitness = neighbor_result.total_fitness
            eval_count += 1

            delta_f = neighbor_fitness - current_fitness

            # Metropolis acceptance criterion (maximizing fitness)
            if delta_f > 0:
                accept = True
            else:
                prob = math.exp(delta_f / max(1e-6, temperature))
                accept = self.rng.random() < prob

            if accept:
                current_solution = neighbor
                current_fitness = neighbor_fitness
                current_result = neighbor_result

                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_individual = current_solution.copy()
                    self.best_result = current_result

            # Cool down
            temperature = max(self.min_temperature, temperature * self.cooling_rate)

            # Record history at interval checkpoints
            if eval_count % step_interval == 0:
                self.history.append({
                    "step": len(self.history),
                    "evaluation": eval_count,
                    "temperature": round(temperature, 3),
                    "current_fitness": round(current_fitness, 3),
                    "best_fitness": round(self.best_fitness, 3),
                    "best_session": self.best_individual.to_tuple_list(),
                    "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                    "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
                })

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
