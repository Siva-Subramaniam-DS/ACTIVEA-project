"""Differential Evolution (DE) benchmark optimization module for ACTIVEA.

Implements Differential Evolution (DE/rand/1/bin) according to Storn & Price (1997)
and Chapter 6 of Eiben & Smith (2015), Introduction to Evolutionary Computing.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator


class DifferentialEvolutionOptimizer:
    """Differential Evolution (DE/rand/1/bin) optimizer for exercise session design.
    
    Uses difference vectors of population members to direct perturbation,
    combining self-referential mutation with binomial crossover and greedy selection.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        pop_size: int = 25,
        max_generations: int = 100,
        f_weight: float = 0.65,  # Differential scaling factor F in [0.5, 0.9]
        cr_rate: float = 0.80,   # Crossover probability CR in [0.6, 0.9]
        inclusion_threshold: float = 5.0,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.f_weight = f_weight
        self.cr_rate = cr_rate
        self.inclusion_threshold = inclusion_threshold
        self.dim = len(self.activities_pool)
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def vector_to_session(self, vec: List[float]) -> ExerciseSession:
        """Decodes continuous D-dimensional vector into an ExerciseSession."""
        active = [(idx, val) for idx, val in enumerate(vec) if val >= self.inclusion_threshold]

        if len(active) < 2:
            ranked = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)
            active = [(ranked[0][0], max(10.0, ranked[0][1])), (ranked[1][0], max(10.0, ranked[1][1]))]
        elif len(active) > 5:
            active = sorted(active, key=lambda x: x[1], reverse=True)[:5]

        session = ExerciseSession()
        for idx, raw_dur in active:
            clamped_dur = round(max(5.0, min(45.0, raw_dur)), 1)
            session.add_activity(self.activities_pool[idx], clamped_dur)
        return session

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs the Differential Evolution optimization."""
        start_time = time.time()
        self.history = []

        # 1. Initialise Population
        population_vectors: List[List[float]] = []
        population_fitness: List[float] = []
        population_results: List[FitnessResult] = []
        population_sessions: List[ExerciseSession] = []

        eval_count = 0

        for _ in range(self.pop_size):
            vec = [0.0] * self.dim
            num_chosen = self.rng.randint(2, 4)
            chosen_indices = self.rng.sample(range(self.dim), num_chosen)
            for idx in chosen_indices:
                vec[idx] = self.rng.uniform(10.0, 30.0)

            sess = self.vector_to_session(vec)
            res = self.evaluator.evaluate(sess)
            fit = res.total_fitness
            eval_count += 1

            population_vectors.append(vec)
            population_fitness.append(fit)
            population_results.append(res)
            population_sessions.append(sess)

        best_idx = max(range(self.pop_size), key=lambda i: population_fitness[i])
        self.best_fitness = population_fitness[best_idx]
        self.best_individual = population_sessions[best_idx].copy()
        self.best_result = population_results[best_idx]

        mean_fit = sum(population_fitness) / len(population_fitness)
        self.history.append({
            "generation": 0,
            "evaluation": eval_count,
            "best_fitness": round(self.best_fitness, 3),
            "mean_fitness": round(mean_fit, 3),
            "best_session": self.best_individual.to_tuple_list(),
            "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
            "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
        })

        # 2. Main DE Generational Loop
        for gen in range(1, self.max_generations + 1):
            for i in range(self.pop_size):
                # Pick 3 random distinct individuals distinct from i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.sample(candidates, 3)

                v_r1 = population_vectors[r1]
                v_r2 = population_vectors[r2]
                v_r3 = population_vectors[r3]
                target = population_vectors[i]

                # Binomial Crossover & Mutation
                trial = [0.0] * self.dim
                j_rand = self.rng.randrange(self.dim)

                for j in range(self.dim):
                    if self.rng.random() < self.cr_rate or j == j_rand:
                        # Mutant gene: v = r1 + F*(r2 - r3)
                        mut_val = v_r1[j] + self.f_weight * (v_r2[j] - v_r3[j])
                        # Bound handling with reflection
                        if mut_val < 0.0:
                            mut_val = abs(mut_val) % 45.0
                        elif mut_val > 45.0:
                            mut_val = 45.0 - (mut_val - 45.0) % 45.0
                        trial[j] = max(0.0, min(45.0, mut_val))
                    else:
                        trial[j] = target[j]

                # Evaluate trial vector
                trial_session = self.vector_to_session(trial)
                trial_res = self.evaluator.evaluate(trial_session)
                trial_fit = trial_res.total_fitness
                eval_count += 1

                # Greedy 1-to-1 replacement
                if trial_fit >= population_fitness[i]:
                    population_vectors[i] = trial
                    population_fitness[i] = trial_fit
                    population_results[i] = trial_res
                    population_sessions[i] = trial_session

                    if trial_fit > self.best_fitness:
                        self.best_fitness = trial_fit
                        self.best_individual = trial_session.copy()
                        self.best_result = trial_res

            mean_fit = sum(population_fitness) / len(population_fitness)
            self.history.append({
                "generation": gen,
                "evaluation": eval_count,
                "best_fitness": round(self.best_fitness, 3),
                "mean_fitness": round(mean_fit, 3),
                "best_session": self.best_individual.to_tuple_list(),
                "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
            })

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
