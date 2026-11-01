"""Random Search baseline optimization module for ACTIVEA benchmarking.

Implements pure random search as the fundamental lower-bound baseline,
testing the extreme exploration limit without memory or exploitation,
as framed in the Computational Intelligence lecture slides on Local Search.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import create_random_session


class RandomSearchOptimizer:
    """Pure Random Search benchmark optimizer.
    
    Serves as the empirical lower-bound control: uniformly samples the search
    space under identical evaluation budgets without memory or gradient guidance.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        total_evaluations: int = 2500,
        checkpoint_interval: int = 50,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.total_evaluations = total_evaluations
        self.checkpoint_interval = checkpoint_interval
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs the Random Search benchmark."""
        start_time = time.time()
        self.history = []

        eval_count = 0
        window_fitnesses: List[float] = []

        while eval_count < self.total_evaluations:
            candidate = create_random_session(self.activities_pool, rng=self.rng)
            result = self.evaluator.evaluate(candidate)
            fitness = result.total_fitness
            eval_count += 1
            window_fitnesses.append(fitness)

            if self.best_individual is None or fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = candidate.copy()
                self.best_result = result

            if eval_count % self.checkpoint_interval == 0 or eval_count == self.total_evaluations:
                mean_fit = sum(window_fitnesses) / len(window_fitnesses) if window_fitnesses else fitness
                self.history.append({
                    "step": len(self.history),
                    "evaluation": eval_count,
                    "generation": eval_count // self.checkpoint_interval,
                    "best_fitness": round(self.best_fitness, 3),
                    "mean_fitness": round(mean_fit, 3),
                    "best_session": self.best_individual.to_tuple_list(),
                    "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                    "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
                })
                window_fitnesses = []

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
