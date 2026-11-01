"""ACTIVEA - Extended Metaheuristic Optimization Algorithms Suite
Coupled with the Champion Machine Learning Caloric Regressor ('Model/best_calorie_model.joblib').

Implements the Full Extended Algorithm Suite for Coursework Tasks 3, 4 & 7:
1. Standard Genetic Algorithm (GA) - Foundational Evolutionary Algorithm (Task 3)
2. Simulated Annealing (SA) - Physics-inspired Trajectory Optimization (Task 4)
3. Particle Swarm Optimization (PSO) - Swarm Intelligence continuous trajectory
4. Differential Evolution (DE) - Difference-vector continuous evolutionary search
5. Iterated Local Search (ILS) - Steepest ascent local search with stochastic kicks
6. Random Search (RS) - Pure uniform exploration (Empirical baseline)
7. Bespoke Repair GA (Repair GA) - Domain-aware heuristic repair evolutionary algorithm (Task 7)

All algorithms natively optimize workout sessions using the trained ML model via MLSavedModelFitnessEvaluator.
"""

from __future__ import annotations
import math
import time
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from evaluate_with_model import EXERCISE_POOL, MLSavedModelFitnessEvaluator

POOL_NAMES: List[str] = sorted(list(EXERCISE_POOL.keys()))
NUM_ACTIVITIES: int = len(POOL_NAMES)
NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(POOL_NAMES)}
IDX_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(POOL_NAMES)}


# =============================================================================
# HELPER FUNCTIONS & OPERATORS
# =============================================================================

def clip_duration(duration: float) -> float:
    """Clips duration to valid coursework range [5.0, 45.0] minutes."""
    return max(5.0, min(45.0, round(float(duration), 1)))


def dedupe_session(session: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Merges duplicate activities by summing their durations, keeping valid order."""
    totals: Dict[str, float] = {}
    order: List[str] = []
    for name, dur in session:
        if name not in totals:
            order.append(name)
            totals[name] = 0.0
        totals[name] += dur
    return [(name, clip_duration(totals[name])) for name in order]


def random_session(rng: random.Random) -> List[Tuple[str, float]]:
    """Generates a random valid session with 2 to 4 activities."""
    k = rng.randint(2, 4)
    chosen = rng.sample(POOL_NAMES, k)
    return [(name, clip_duration(rng.uniform(8.0, 30.0))) for name in chosen]


def mutate_session(session: List[Tuple[str, float]], rng: random.Random, rate: float = 0.25) -> List[Tuple[str, float]]:
    """Perturbs a session via duration adjustment, activity swap, addition, or removal."""
    mutated = [[name, dur] for name, dur in session]

    # 1. Duration adjustments
    for item in mutated:
        if rng.random() < rate:
            delta = rng.choice([-10.0, -5.0, -2.0, 2.0, 5.0, 10.0])
            item[1] = clip_duration(item[1] + delta)

    # 2. Activity swap
    if mutated and rng.random() < rate:
        idx = rng.randrange(len(mutated))
        used = {item[0] for item in mutated}
        avail = [name for name in POOL_NAMES if name not in used]
        if avail:
            mutated[idx][0] = rng.choice(avail)

    # 3. Activity addition
    if len(mutated) < 5 and rng.random() < rate:
        used = {item[0] for item in mutated}
        avail = [name for name in POOL_NAMES if name not in used]
        if avail:
            mutated.append([rng.choice(avail), clip_duration(rng.uniform(8.0, 20.0))])

    # 4. Activity deletion
    if len(mutated) > 2 and rng.random() < rate:
        mutated.pop(rng.randrange(len(mutated)))

    # Dedupe and ensure size between 2 and 5
    clean = dedupe_session([(name, dur) for name, dur in mutated])
    while len(clean) < 2:
        used = {name for name, _ in clean}
        avail = [name for name in POOL_NAMES if name not in used]
        clean.append((rng.choice(avail), clip_duration(rng.uniform(8.0, 20.0))))
    if len(clean) > 5:
        clean = rng.sample(clean, 5)
    return clean


def vector_to_session(vector: np.ndarray, threshold: float = 5.0) -> List[Tuple[str, float]]:
    """Converts a continuous 10D duration vector into an active session tuple list."""
    items = []
    for i, dur in enumerate(vector):
        if dur >= threshold:
            items.append((IDX_TO_NAME[i], clip_duration(dur)))
    
    # Enforce coursework constraints: between 2 and 5 activities
    if len(items) < 2:
        top_indices = np.argsort(vector)[-2:]
        items = [(IDX_TO_NAME[i], clip_duration(max(threshold, vector[i]))) for i in top_indices]
    elif len(items) > 5:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:5]
    return items


def session_to_vector(session: List[Tuple[str, float]]) -> np.ndarray:
    """Encodes an activity-duration tuple list into a 10D continuous vector."""
    vec = np.zeros(NUM_ACTIVITIES, dtype=np.float64)
    for name, dur in session:
        if name in NAME_TO_IDX:
            vec[NAME_TO_IDX[name]] += dur
    return vec


# =============================================================================
# 1. RANDOM SEARCH (EMPIRICAL BASELINE)
# =============================================================================

class RandomSearchML:
    """Pure random uniform stochastic search serving as empirical baseline lower-bound."""
    def __init__(self, evaluator: MLSavedModelFitnessEvaluator, seed: int = 42):
        self.evaluator = evaluator
        self.rng = random.Random(seed)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        best_sol = random_session(self.rng)
        best_fit, best_det = self.evaluator.evaluate(best_sol)
        
        history = [{"eval": 1, "best_fitness": best_fit}]
        for eval_idx in range(2, budget + 1):
            cand = random_session(self.rng)
            fit, det = self.evaluator.evaluate(cand)
            if fit > best_fit:
                best_fit = fit
                best_sol = cand
                best_det = det
            if eval_idx % checkpoint_interval == 0 or eval_idx == budget:
                history.append({"eval": eval_idx, "best_fitness": best_fit})

        return {
            "algorithm": "Random Search",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": budget,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 2. SIMULATED ANNEALING (TASK 4 NATURE-INSPIRED TRAJECTORY BENCHMARK)
# =============================================================================

class SimulatedAnnealingML:
    """Task 4: Simulated Annealing (Kirkpatrick et al., 1983; Lecture 'Local Search').
    Uses continuous geometric cooling and Metropolis acceptance criterion.
    """
    def __init__(
        self,
        evaluator: MLSavedModelFitnessEvaluator,
        t_initial: float = 40.0,
        t_final: float = 0.05,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.t_initial = t_initial
        self.t_final = t_final
        self.rng = random.Random(seed)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        # Cooling decay factor per evaluation
        decay = (self.t_final / self.t_initial) ** (1.0 / max(1, budget - 1))

        current_sol = random_session(self.rng)
        current_fit, current_det = self.evaluator.evaluate(current_sol)
        best_sol, best_fit, best_det = current_sol, current_fit, current_det

        temperature = self.t_initial
        history = [{"eval": 1, "best_fitness": best_fit}]

        for eval_idx in range(2, budget + 1):
            neighbor = mutate_session(current_sol, self.rng, rate=0.25)
            neighbor_fit, neighbor_det = self.evaluator.evaluate(neighbor)

            delta = neighbor_fit - current_fit
            # Metropolis criterion
            if delta > 0.0 or self.rng.random() < math.exp(delta / max(1e-6, temperature)):
                current_sol = neighbor
                current_fit = neighbor_fit
                current_det = neighbor_det

            if current_fit > best_fit:
                best_fit = current_fit
                best_sol = current_sol
                best_det = current_det

            temperature = max(self.t_final, temperature * decay)

            if eval_idx % checkpoint_interval == 0 or eval_idx == budget:
                history.append({"eval": eval_idx, "best_fitness": best_fit})

        return {
            "algorithm": "Simulated Annealing",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": budget,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 3. ITERATED LOCAL SEARCH (LOCAL SEARCH WITH KICKS)
# =============================================================================

class IteratedLocalSearchML:
    """Iterated Local Search: Combines greedy steepest-ascent local search with
    periodic stochastic perturbation ('kicks') to escape local optima.
    """
    def __init__(self, evaluator: MLSavedModelFitnessEvaluator, kick_strength: int = 3, seed: int = 42):
        self.evaluator = evaluator
        self.kick_strength = kick_strength
        self.rng = random.Random(seed)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        current_sol = random_session(self.rng)
        current_fit, current_det = self.evaluator.evaluate(current_sol)
        best_sol, best_fit, best_det = current_sol, current_fit, current_det

        history = [{"eval": 1, "best_fitness": best_fit}]
        stagnation = 0

        for eval_idx in range(2, budget + 1):
            if stagnation >= 25:
                # Strong perturbation 'kick' on stagnation
                cand = mutate_session(current_sol, self.rng, rate=0.60)
                stagnation = 0
            else:
                # Local exploitation step
                cand = mutate_session(current_sol, self.rng, rate=0.20)

            fit, det = self.evaluator.evaluate(cand)
            if fit >= current_fit:
                current_sol = cand
                current_fit = fit
                current_det = det
                stagnation = 0
            else:
                stagnation += 1

            if current_fit > best_fit:
                best_fit = current_fit
                best_sol = current_sol
                best_det = current_det

            if eval_idx % checkpoint_interval == 0 or eval_idx == budget:
                history.append({"eval": eval_idx, "best_fitness": best_fit})

        return {
            "algorithm": "Iterated Local Search",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": budget,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 4. PARTICLE SWARM OPTIMIZATION (SWARM INTELLIGENCE)
# =============================================================================

class ParticleSwarmML:
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995; Clerc constriction).
    Continuous swarm trajectories over 10D activity-duration allocations.
    """
    def __init__(
        self,
        evaluator: MLSavedModelFitnessEvaluator,
        swarm_size: int = 25,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        v_max: float = 10.0,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.rng = random.Random(seed)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        # Initialize swarm particles
        positions = []
        velocities = []
        p_bests = []
        p_best_fits = []

        eval_count = 0
        best_sol = None
        best_fit = -1.0
        best_det = None
        history = []

        for _ in range(self.swarm_size):
            init_s = random_session(self.rng)
            pos = session_to_vector(init_s)
            vel = np.array([self.rng.uniform(-2.0, 2.0) for _ in range(NUM_ACTIVITIES)])
            fit, det = self.evaluator.evaluate(init_s)
            eval_count += 1

            positions.append(pos)
            velocities.append(vel)
            p_bests.append(pos.copy())
            p_best_fits.append(fit)

            if fit > best_fit:
                best_fit = fit
                best_sol = init_s
                best_det = det

        history.append({"eval": eval_count, "best_fitness": best_fit})
        g_best = session_to_vector(best_sol)

        while eval_count < budget:
            for i in range(self.swarm_size):
                if eval_count >= budget:
                    break

                r1 = np.array([self.rng.random() for _ in range(NUM_ACTIVITIES)])
                r2 = np.array([self.rng.random() for _ in range(NUM_ACTIVITIES)])

                # Velocity update
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (p_bests[i] - positions[i]) +
                    self.c2 * r2 * (g_best - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                # Position update & bounding
                positions[i] = np.clip(positions[i] + velocities[i], 0.0, 45.0)

                cand_s = vector_to_session(positions[i])
                fit, det = self.evaluator.evaluate(cand_s)
                eval_count += 1

                if fit > p_best_fits[i]:
                    p_best_fits[i] = fit
                    p_bests[i] = positions[i].copy()

                if fit > best_fit:
                    best_fit = fit
                    best_sol = cand_s
                    best_det = det
                    g_best = positions[i].copy()

                if eval_count % checkpoint_interval == 0 or eval_count == budget:
                    history.append({"eval": eval_count, "best_fitness": best_fit})

        return {
            "algorithm": "Particle Swarm Optimization",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": eval_count,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 5. DIFFERENTIAL EVOLUTION (DE/rand/1/bin)
# =============================================================================

class DifferentialEvolutionML:
    """Differential Evolution (Storn & Price, 1997; Eiben & Smith, Ch. 6).
    Continuous self-referential vector differences guide mutation and crossover.
    """
    def __init__(
        self,
        evaluator: MLSavedModelFitnessEvaluator,
        pop_size: int = 25,
        f_scale: float = 0.5,
        cr_prob: float = 0.8,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.f_scale = f_scale
        self.cr_prob = cr_prob
        self.rng = random.Random(seed)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        population = []
        fits = []
        eval_count = 0
        best_sol = None
        best_fit = -1.0
        best_det = None
        history = []

        for _ in range(self.pop_size):
            init_s = random_session(self.rng)
            vec = session_to_vector(init_s)
            fit, det = self.evaluator.evaluate(init_s)
            eval_count += 1

            population.append(vec)
            fits.append(fit)

            if fit > best_fit:
                best_fit = fit
                best_sol = init_s
                best_det = det

        history.append({"eval": eval_count, "best_fitness": best_fit})

        while eval_count < budget:
            for i in range(self.pop_size):
                if eval_count >= budget:
                    break

                # Pick 3 distinct individuals different from i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.sample(candidates, 3)

                # Donor vector via differential mutation
                donor = population[r1] + self.f_scale * (population[r2] - population[r3])
                donor = np.clip(donor, 0.0, 45.0)

                # Binomial crossover
                trial = population[i].copy()
                rand_j = self.rng.randrange(NUM_ACTIVITIES)
                for j in range(NUM_ACTIVITIES):
                    if j == rand_j or self.rng.random() < self.cr_prob:
                        trial[j] = donor[j]

                cand_s = vector_to_session(trial)
                trial_fit, trial_det = self.evaluator.evaluate(cand_s)
                eval_count += 1

                # Greedy selection
                if trial_fit >= fits[i]:
                    population[i] = trial
                    fits[i] = trial_fit

                if trial_fit > best_fit:
                    best_fit = trial_fit
                    best_sol = cand_s
                    best_det = trial_det

                if eval_count % checkpoint_interval == 0 or eval_count == budget:
                    history.append({"eval": eval_count, "best_fitness": best_fit})

        return {
            "algorithm": "Differential Evolution",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": eval_count,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 6. BESPOKE REPAIR GA (TASK 7 DOMAIN & CALORIC REPAIR HEURISTICS)
# =============================================================================

class BespokeRepairGA_ML:
    """Task 7: Advanced Bespoke Repair Genetic Algorithm.
    Integrates physiological domain-gap repair and caloric target scaling heuristics.
    """
    def __init__(
        self,
        evaluator: MLSavedModelFitnessEvaluator,
        pop_size: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)

    def repair(self, session: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Heuristically repairs missing clinical fitness domains and caloric imbalances."""
        session = dedupe_session(session)
        covered_domains = set().union(*[EXERCISE_POOL[name].domains for name, _ in session])

        # 1. Domain repair: Ensure at least 5 domains are covered
        all_required = {"CV", "ST", "ME", "FL", "BA", "FM"}
        missing = all_required - covered_domains
        if missing and len(session) < 5:
            # Find an activity in pool that covers the missing domain
            candidates = [name for name in POOL_NAMES if any(d in missing for d in EXERCISE_POOL[name].domains)]
            if candidates:
                session.append((self.rng.choice(candidates), clip_duration(self.rng.uniform(8.0, 15.0))))

        # 2. Duration balance repair
        tot_dur = sum(dur for _, dur in session)
        if tot_dur > 70.0:
            scale = 65.0 / tot_dur
            session = [(name, clip_duration(dur * scale)) for name, dur in session]
        elif tot_dur < 25.0:
            scale = 30.0 / max(1.0, tot_dur)
            session = [(name, clip_duration(dur * scale)) for name, dur in session]

        return dedupe_session(session)

    def crossover(self, p1: List[Tuple[str, float]], p2: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        if self.rng.random() >= self.crossover_rate:
            return list(p1), list(p2)
        merged = {}
        for name, dur in p1 + p2:
            merged[name] = merged.get(name, 0.0) + dur
        order = list(merged.keys())
        self.rng.shuffle(order)
        if len(order) < 2:
            return list(p1), list(p2)
        split = self.rng.randint(1, len(order) - 1)
        c1 = [(name, merged[name]) for name in order[:split]]
        c2 = [(name, merged[name]) for name in order[split:]]
        return self.repair(c1), self.repair(c2)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        population = [self.repair(random_session(self.rng)) for _ in range(self.pop_size)]
        fits = []
        eval_count = 0
        best_sol = None
        best_fit = -1.0
        best_det = None
        history = []

        for ind in population:
            fit, det = self.evaluator.evaluate(ind)
            eval_count += 1
            fits.append(fit)
            if fit > best_fit:
                best_fit = fit
                best_sol = ind
                best_det = det
            if eval_count >= budget:
                break

        history.append({"eval": eval_count, "best_fitness": best_fit})

        while eval_count < budget:
            # Elitism: top 2
            ranked = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            elites = [population[ranked[0]], population[ranked[1]]]

            offspring = []
            while len(offspring) < (self.pop_size - 2) and eval_count < budget:
                # Tournament
                p1_idx = max(self.rng.sample(range(len(population)), min(self.tournament_size, len(population))), key=lambda i: fits[i])
                p2_idx = max(self.rng.sample(range(len(population)), min(self.tournament_size, len(population))), key=lambda i: fits[i])
                c1, c2 = self.crossover(population[p1_idx], population[p2_idx])

                for child in (c1, c2):
                    if len(offspring) < (self.pop_size - 2) and eval_count < budget:
                        if self.rng.random() < self.mutation_rate:
                            child = self.repair(mutate_session(child, self.rng, rate=0.20))
                        offspring.append(child)

            # Evaluate new offspring
            new_fits = []
            for child in offspring:
                if eval_count >= budget:
                    break
                fit, det = self.evaluator.evaluate(child)
                eval_count += 1
                new_fits.append(fit)
                if fit > best_fit:
                    best_fit = fit
                    best_sol = child
                    best_det = det

            population = elites + offspring[:len(new_fits)]
            fits = [fits[ranked[0]], fits[ranked[1]]] + new_fits

            if eval_count % checkpoint_interval == 0 or eval_count >= budget:
                history.append({"eval": eval_count, "best_fitness": best_fit})

        return {
            "algorithm": "Bespoke Repair GA",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": eval_count,
            "runtime_sec": time.time() - t0
        }


# =============================================================================
# 7. STANDARD GENETIC ALGORITHM (TASK 3 REFERENCE BASELINE)
# =============================================================================

class StandardGeneticAlgorithmML:
    """Task 3: Standard Generational Genetic Algorithm with Tournament Selection,
    uniform crossover, multi-mode mutation, and elitism.
    """
    def __init__(
        self,
        evaluator: MLSavedModelFitnessEvaluator,
        pop_size: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        seed: int = 42
    ):
        self.evaluator = evaluator
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)

    def crossover(self, p1: List[Tuple[str, float]], p2: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        if self.rng.random() >= self.crossover_rate:
            return list(p1), list(p2)
        merged = {}
        for name, dur in p1 + p2:
            merged[name] = merged.get(name, 0.0) + dur
        order = list(merged.keys())
        self.rng.shuffle(order)
        if len(order) < 2:
            return list(p1), list(p2)
        split = self.rng.randint(1, len(order) - 1)
        c1 = [(name, merged[name]) for name in order[:split]]
        c2 = [(name, merged[name]) for name in order[split:]]
        return dedupe_session(c1), dedupe_session(c2)

    def optimize(self, budget: int = 2500, checkpoint_interval: int = 50) -> Dict[str, Any]:
        t0 = time.time()
        population = [random_session(self.rng) for _ in range(self.pop_size)]
        fits = []
        eval_count = 0
        best_sol = None
        best_fit = -1.0
        best_det = None
        history = []

        for ind in population:
            fit, det = self.evaluator.evaluate(ind)
            eval_count += 1
            fits.append(fit)
            if fit > best_fit:
                best_fit = fit
                best_sol = ind
                best_det = det
            if eval_count >= budget:
                break

        history.append({"eval": eval_count, "best_fitness": best_fit})

        while eval_count < budget:
            ranked = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            elites = [population[ranked[0]], population[ranked[1]]]

            offspring = []
            while len(offspring) < (self.pop_size - 2) and eval_count < budget:
                p1_idx = max(self.rng.sample(range(len(population)), min(self.tournament_size, len(population))), key=lambda i: fits[i])
                p2_idx = max(self.rng.sample(range(len(population)), min(self.tournament_size, len(population))), key=lambda i: fits[i])
                c1, c2 = self.crossover(population[p1_idx], population[p2_idx])

                for child in (c1, c2):
                    if len(offspring) < (self.pop_size - 2) and eval_count < budget:
                        if self.rng.random() < self.mutation_rate:
                            child = mutate_session(child, self.rng, rate=0.20)
                        offspring.append(child)

            new_fits = []
            for child in offspring:
                if eval_count >= budget:
                    break
                fit, det = self.evaluator.evaluate(child)
                eval_count += 1
                new_fits.append(fit)
                if fit > best_fit:
                    best_fit = fit
                    best_sol = child
                    best_det = det

            population = elites + offspring[:len(new_fits)]
            fits = [fits[ranked[0]], fits[ranked[1]]] + new_fits

            if eval_count % checkpoint_interval == 0 or eval_count >= budget:
                history.append({"eval": eval_count, "best_fitness": best_fit})

        return {
            "algorithm": "Standard Genetic Algorithm",
            "best_solution": best_sol,
            "best_fitness": best_fit,
            "best_details": best_det,
            "history": pd.DataFrame(history),
            "eval_count": eval_count,
            "runtime_sec": time.time() - t0
        }
