"""Core ACTIVEA evolutionary algorithm (Task 3).

The loop is the generational evolutionary algorithm on the subject slides
``Subject PPt/Evolutionary Algorithms Knowledge Acquisition 2H (1).pdf``:
initialise, evaluate, select parents, recombine, mutate, select survivors,
and test termination every generation. The last slide names the source:
A. E. Eiben and J. E. Smith, Introduction to Evolutionary Computing,
chapters 2 to 5.
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, FitnessResult
from .fitness import ActiveaFitnessEvaluator
from .operators import (
    initialize_population,
    tournament_selection,
    roulette_selection,
    uniform_crossover,
    single_point_crossover,
    standard_mutation,
)


class ActiveaGeneticAlgorithm:
    """Generational genetic algorithm for exercise-session search.

    Subject slides, Evolutionary Algorithms Knowledge Acquisition:
    parent selection, recombination, mutation, then survivor selection,
    repeated until termination. Binary strings are the genetic-algorithm
    representation on those slides; this search uses the same loop on
    (activity, duration) genes. Elitism keeps the fittest solutions, which
    the slides state as "always keep at least one copy of the fittest
    solution so far". The recommended reading on the final slide is
    Eiben and Smith, Introduction to Evolutionary Computing, chapters 2-5.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        pop_size: int = 50,
        max_generations: int = 60,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 2,
        min_quality_threshold: Optional[float] = None,
        selection_type: str = "tournament",  # "tournament" or "roulette"
        crossover_type: str = "uniform",     # "uniform" or "single_point"
        repair_operator: Optional[Any] = None,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.min_quality_threshold = min_quality_threshold
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.repair_operator = repair_operator
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # State tracking
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def calculate_population_diversity(self, population: List[ExerciseSession]) -> float:
        """Measures phenotypic diversity across the population based on domain activity compositions."""
        if len(population) <= 1:
            return 0.0

        all_names = sorted({a.name for a in self.activities_pool})
        name_to_idx = {name: i for i, name in enumerate(all_names)}

        # Vector of durations per activity in pool
        vectors: List[List[float]] = []
        for ind in population:
            vec = [0.0] * len(all_names)
            for item in ind.items:
                if item.activity.name in name_to_idx:
                    vec[name_to_idx[item.activity.name]] += item.duration
            vectors.append(vec)

        # Average pairwise Euclidean distance
        pair_distances = []
        sample_size = min(30, len(population))
        sampled_indices = self.rng.sample(range(len(population)), sample_size)
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                v1 = vectors[sampled_indices[i]]
                v2 = vectors[sampled_indices[j]]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
                pair_distances.append(dist)

        return sum(pair_distances) / len(pair_distances) if pair_distances else 0.0

    def evolve(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Run the subject-slide evolutionary loop until termination."""
        start_time = time.time()
        self.history = []

        # Initialisation is random (subject slides: Initialisation / Termination).
        population = initialize_population(
            pop_size=self.pop_size,
            activities_pool=self.activities_pool,
            rng=self.rng,
        )

        # Evaluate initial population
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
            "best_calories": self.best_result.details.get("total_calories", 0.0),
            "best_duration": self.best_result.details.get("total_duration", 0.0),
        })

        # Termination is checked every generation: generation limit, or a
        # desired fitness when one is set (subject slides: Initialisation / Termination).
        while gen < self.max_generations:
            if self.min_quality_threshold is not None and self.best_fitness >= self.min_quality_threshold:
                break
            # Elitism, then the generational replacement below. The slides say
            # always keep a copy of the fittest solution, and that a genetic
            # algorithm replaces the parents with the offspring.
            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_values[i], reverse=True)
            elites = [population[i].copy() for i in sorted_indices[: self.elitism_count]]

            # Parent selection. Default is tournament: pick k at random and
            # keep the best (subject slides: Parent selection by tournament).
            selected_parents: List[ExerciseSession] = []
            for _ in range(self.pop_size - self.elitism_count):
                if self.selection_type == "roulette":
                    p = roulette_selection(population, fitness_values, rng=self.rng)
                else:
                    p = tournament_selection(population, fitness_values, k=self.tournament_size, rng=self.rng)
                selected_parents.append(p)

            # Variation. Crossover takes two parents and returns two children.
            # Mutation takes one child and returns one child (subject slides:
            # Variation operators).
            offspring: List[ExerciseSession] = []
            for i in range(0, len(selected_parents), 2):
                p1 = selected_parents[i]
                p2 = selected_parents[(i + 1) % len(selected_parents)]

                if self.rng.random() < self.crossover_rate:
                    if self.crossover_type == "single_point":
                        c1, c2 = single_point_crossover(p1, p2, rng=self.rng)
                    else:
                        c1, c2 = uniform_crossover(p1, p2, rng=self.rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if self.repair_operator is not None:
                    c1 = self.repair_operator.repair_and_mutate(c1, self.mutation_rate, rng=self.rng)
                    c2 = self.repair_operator.repair_and_mutate(c2, self.mutation_rate, rng=self.rng)
                else:
                    c1 = standard_mutation(c1, self.activities_pool, self.mutation_rate, rng=self.rng)
                    c2 = standard_mutation(c2, self.activities_pool, self.mutation_rate, rng=self.rng)

                offspring.extend([c1, c2])

            # Survivor selection: generational model plus the elite copies.
            population = elites + offspring[: self.pop_size - self.elitism_count]

            # Fitness is one real value per session and is maximised
            # (subject slides: Evaluation (fitness) function).
            fitness_results = [self.evaluator.evaluate(ind) for ind in population]
            fitness_values = [res.total_fitness for res in fitness_results]

            current_best_idx = max(range(len(population)), key=lambda i: fitness_values[i])
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
