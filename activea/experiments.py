"""Experimentation and evaluation module for ACTIVEA (Task 5 & Task 6).

Implements:
1. Task 5: Systematic parameter tuning experiments (population size, crossover/mutation rates, tournament size).
2. Task 6: Diversity preservation via Deterministic Crowding and Fitness Sharing niching.
"""

from __future__ import annotations
import copy
import math
import random
import time
from typing import List, Dict, Any, Tuple, Optional

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import (
    initialize_population,
    tournament_selection,
    uniform_crossover,
    standard_mutation,
)
from .core_ga import ActiveaGeneticAlgorithm


def calculate_session_distance(s1: ExerciseSession, s2: ExerciseSession, pool_names: List[str]) -> float:
    """Calculates Euclidean phenotypic distance between two exercise sessions."""
    d1 = {it.activity.name: it.duration for it in s1.items}
    d2 = {it.activity.name: it.duration for it in s2.items}
    return math.sqrt(sum((d1.get(name, 0.0) - d2.get(name, 0.0)) ** 2 for name in pool_names))


class CrowdingGeneticAlgorithm(ActiveaGeneticAlgorithm):
    """Task 6: Diversity-Preserving Evolutionary Algorithm using Deterministic Crowding.
    
    In Deterministic Crowding:
    1. Parents are paired and produce offspring.
    2. Offspring compete in tournaments against their closest phenotypic parents.
    3. Prevents premature convergence, preserving niches of diverse workout regimes.
    """

    def evolve(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        start_time = time.time()
        self.history = []
        pool_names = sorted({a.name for a in self.activities_pool})

        population = initialize_population(
            pop_size=self.pop_size,
            activities_pool=self.activities_pool,
            rng=self.rng,
        )

        fitness_results = [self.evaluator.evaluate(ind) for ind in population]
        fitness_values = [res.total_fitness for res in fitness_results]

        best_idx = max(range(len(population)), key=lambda i: fitness_values[i])
        self.best_individual = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.best_result = fitness_results[best_idx]

        gen = 0
        diversity = self.calculate_population_diversity(population)
        mean_fit = sum(fitness_values) / len(fitness_values)

        self.history.append({
            "generation": 0,
            "best_fitness": self.best_fitness,
            "mean_fitness": round(mean_fit, 3),
            "diversity": round(diversity, 3),
            "best_session": self.best_individual.to_tuple_list(),
            "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
            "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
        })

        while gen < self.max_generations:
            if self.min_quality_threshold is not None and self.best_fitness >= self.min_quality_threshold:
                break
            # Deterministic Crowding: Shuffle population pairs
            shuffled_indices = list(range(len(population)))
            self.rng.shuffle(shuffled_indices)

            for i in range(0, len(shuffled_indices) - 1, 2):
                p1_idx = shuffled_indices[i]
                p2_idx = shuffled_indices[i + 1]
                p1 = population[p1_idx]
                p2 = population[p2_idx]

                # Crossover
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = uniform_crossover(p1, p2, rng=self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                c1 = standard_mutation(c1, self.activities_pool, self.mutation_rate, rng=self.rng)
                c2 = standard_mutation(c2, self.activities_pool, self.mutation_rate, rng=self.rng)

                f_c1 = self.evaluator.evaluate(c1)
                f_c2 = self.evaluator.evaluate(c2)

                # Pairwise distance for parent-child matching
                d_p1_c1 = calculate_session_distance(p1, c1, pool_names)
                d_p2_c2 = calculate_session_distance(p2, c2, pool_names)
                d_p1_c2 = calculate_session_distance(p1, c2, pool_names)
                d_p2_c1 = calculate_session_distance(p2, c1, pool_names)

                if (d_p1_c1 + d_p2_c2) <= (d_p1_c2 + d_p2_c1):
                    # Tournament 1: p1 vs c1
                    if f_c1.total_fitness > fitness_values[p1_idx]:
                        population[p1_idx] = c1
                        fitness_values[p1_idx] = f_c1.total_fitness
                        fitness_results[p1_idx] = f_c1
                    # Tournament 2: p2 vs c2
                    if f_c2.total_fitness > fitness_values[p2_idx]:
                        population[p2_idx] = c2
                        fitness_values[p2_idx] = f_c2.total_fitness
                        fitness_results[p2_idx] = f_c2
                else:
                    # Tournament 1: p1 vs c2
                    if f_c2.total_fitness > fitness_values[p1_idx]:
                        population[p1_idx] = c2
                        fitness_values[p1_idx] = f_c2.total_fitness
                        fitness_results[p1_idx] = f_c2
                    # Tournament 2: p2 vs c1
                    if f_c1.total_fitness > fitness_values[p2_idx]:
                        population[p2_idx] = c1
                        fitness_values[p2_idx] = f_c1.total_fitness
                        fitness_results[p2_idx] = f_c1

            current_best_idx = max(range(len(population)), key=lambda idx: fitness_values[idx])
            if fitness_values[current_best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[current_best_idx]
                self.best_individual = population[current_best_idx].copy()
                self.best_result = fitness_results[current_best_idx]

            gen += 1
            mean_fit = sum(fitness_values) / len(fitness_values)
            diversity = self.calculate_population_diversity(population)

            self.history.append({
                "generation": gen,
                "best_fitness": round(self.best_fitness, 3),
                "mean_fitness": round(mean_fit, 3),
                "diversity": round(diversity, 3),
                "best_session": self.best_individual.to_tuple_list(),
                "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
            })

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history


class ParameterTuningExperiment:
    """Task 5: Systematic parameter sensitivity evaluation."""

    def __init__(self, activities_pool: Dict[str, Activity] | List[Activity]):
        self.activities_pool = activities_pool

    def run_grid_experiment(
        self,
        pop_sizes: List[int] = [20, 50, 100],
        mutation_rates: List[float] = [0.05, 0.15, 0.30],
        crossover_rates: List[float] = [0.6, 0.8, 0.95],
        tournament_sizes: List[int] = [2, 3, 5],
        num_seeds: int = 5,
        generations: int = 40,
    ) -> List[Dict[str, Any]]:
        """Evaluates configurations across multiple random seeds."""
        results: List[Dict[str, Any]] = []

        # Focused parameter sweep
        configs = []
        # Sweep 1: Population size
        for ps in pop_sizes:
            configs.append({"pop_size": ps, "mutation_rate": 0.15, "crossover_rate": 0.8, "tourn_size": 3, "label": f"PopSize_{ps}"})
        # Sweep 2: Mutation rate
        for mr in mutation_rates:
            if mr != 0.15:
                configs.append({"pop_size": 50, "mutation_rate": mr, "crossover_rate": 0.8, "tourn_size": 3, "label": f"MutRate_{mr}"})
        # Sweep 3: Crossover rate
        for cr in crossover_rates:
            if cr != 0.8:
                configs.append({"pop_size": 50, "mutation_rate": 0.15, "crossover_rate": cr, "tourn_size": 3, "label": f"CrossRate_{cr}"})
        # Sweep 4: Tournament size
        for ts in tournament_sizes:
            if ts != 3:
                configs.append({"pop_size": 50, "mutation_rate": 0.15, "crossover_rate": 0.8, "tourn_size": ts, "label": f"TournSize_{ts}"})

        for cfg in configs:
            best_fits = []
            durations = []
            exec_times = []

            for seed in range(num_seeds):
                ga = ActiveaGeneticAlgorithm(
                    activities_pool=self.activities_pool,
                    pop_size=cfg["pop_size"],
                    max_generations=generations,
                    crossover_rate=cfg["crossover_rate"],
                    mutation_rate=cfg["mutation_rate"],
                    tournament_size=cfg["tourn_size"],
                    seed=100 + seed,
                )
                best_ind, best_res, _ = ga.evolve()
                best_fits.append(best_res.total_fitness)
                durations.append(best_ind.total_duration)
                exec_times.append(ga.execution_time_sec)

            mean_fit = sum(best_fits) / len(best_fits)
            std_fit = math.sqrt(sum((f - mean_fit) ** 2 for f in best_fits) / len(best_fits)) if len(best_fits) > 1 else 0.0

            results.append({
                "label": cfg["label"],
                "pop_size": cfg["pop_size"],
                "mutation_rate": cfg["mutation_rate"],
                "crossover_rate": cfg["crossover_rate"],
                "tourn_size": cfg["tourn_size"],
                "mean_best_fitness": round(mean_fit, 3),
                "std_best_fitness": round(std_fit, 3),
                "mean_exec_time_sec": round(sum(exec_times) / len(exec_times), 3),
                "mean_session_duration": round(sum(durations) / len(durations), 1),
            })

        return results
