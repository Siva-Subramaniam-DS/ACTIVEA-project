"""Genetic operators module for ACTIVEA (Task 2 & 3).

Implements selection, crossover, mutation, and population generation operators
tailored for the tuple-based chromosome representation: [(activity, duration), ...].
"""

from __future__ import annotations
import copy
import random
from typing import List, Dict, Tuple, Optional

from .models import Activity, ExerciseSession, SessionItem


def create_random_session(
    activities_pool: Dict[str, Activity] | List[Activity],
    min_items: int = 2,
    max_items: int = 4,
    min_dur: float = 10.0,
    max_dur: float = 25.0,
    rng: Optional[random.Random] = None,
) -> ExerciseSession:
    """Creates a random valid exercise session chromosome."""
    r = rng if rng is not None else random.Random()
    if isinstance(activities_pool, dict):
        pool_list = list(activities_pool.values())
    else:
        pool_list = list(activities_pool)

    num_items = r.randint(min_items, max_items)
    selected_acts = r.sample(pool_list, min(num_items, len(pool_list)))

    session = ExerciseSession()
    for act in selected_acts:
        dur = round(r.uniform(min_dur, max_dur), 0)
        session.add_activity(act, dur)
    return session


def initialize_population(
    pop_size: int,
    activities_pool: Dict[str, Activity] | List[Activity],
    min_items: int = 2,
    max_items: int = 4,
    rng: Optional[random.Random] = None,
) -> List[ExerciseSession]:
    """Generates an initial population of diverse exercise sessions."""
    r = rng if rng is not None else random.Random()
    return [
        create_random_session(activities_pool, min_items, max_items, rng=r)
        for _ in range(pop_size)
    ]


# ---------------------------------------------------------
# Selection Operators
# ---------------------------------------------------------

def tournament_selection(
    population: List[ExerciseSession],
    fitness_values: List[float],
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> ExerciseSession:
    """Tournament selection from the subject slides.

    Evolutionary Algorithms Knowledge Acquisition: pick k members at random,
    then select the best of them. The same operator is Section 4.2 of
    Eiben and Smith (2015), the book named on the final slide.
    """
    r = rng if rng is not None else random.Random()
    indices = r.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda idx: fitness_values[idx])
    return population[best_idx].copy()


def roulette_selection(
    population: List[ExerciseSession],
    fitness_values: List[float],
    rng: Optional[random.Random] = None,
) -> ExerciseSession:
    """Fitness-proportionate selection.

    Subject slides, "Parent selection by Roulette wheel": the chance of
    mating is the individual's fitness divided by the population total.
    """
    r = rng if rng is not None else random.Random()
    # Shift minimum fitness to ensure non-negative probabilities
    min_f = min(fitness_values)
    adjusted_f = [max(0.01, f - min_f + 0.1) for f in fitness_values]
    total_f = sum(adjusted_f)
    pick = r.uniform(0.0, total_f)
    current = 0.0
    for ind, fit in zip(population, adjusted_f):
        current += fit
        if current >= pick:
            return ind.copy()
    return population[-1].copy()


# ---------------------------------------------------------
# Crossover Operators
# ---------------------------------------------------------

def single_point_crossover(
    parent1: ExerciseSession,
    parent2: ExerciseSession,
    rng: Optional[random.Random] = None,
) -> Tuple[ExerciseSession, ExerciseSession]:
    """One-point crossover: two parents in, two children out.

    Subject slides, "Recombination: one point crossover". A cut is taken in
    each session and the tails are swapped. Duplicate activities then have
    their durations added, which is the lecture rule for this representation.
    """
    r = rng if rng is not None else random.Random()
    p1_items = parent1.items
    p2_items = parent2.items

    if len(p1_items) <= 1 or len(p2_items) <= 1:
        return parent1.copy(), parent2.copy()

    pt1 = r.randint(1, len(p1_items) - 1)
    pt2 = r.randint(1, len(p2_items) - 1)

    child1_items = [it.copy() for it in p1_items[:pt1]] + [it.copy() for it in p2_items[pt2:]]
    child2_items = [it.copy() for it in p2_items[:pt2]] + [it.copy() for it in p1_items[pt1:]]

    # Deduplicate activities within each child (consolidate durations)
    return _deduplicate_session(ExerciseSession(child1_items)), _deduplicate_session(ExerciseSession(child2_items))


def uniform_crossover(
    parent1: ExerciseSession,
    parent2: ExerciseSession,
    rng: Optional[random.Random] = None,
) -> Tuple[ExerciseSession, ExerciseSession]:
    """Recombination of two sessions into two children.

    Subject slides: crossover takes two inputs and typically produces two
    outputs. The genes are pooled, split, and a repeated activity keeps the
    summed duration (lecture rule for this representation; Eiben and Smith,
    2015, Section 4.4, the book named on the final subject slide).
    """
    r = rng if rng is not None else random.Random()
    all_items = [it.copy() for it in parent1.items] + [it.copy() for it in parent2.items]
    r.shuffle(all_items)

    mid = max(2, len(all_items) // 2)
    child1 = _deduplicate_session(ExerciseSession(all_items[:mid]))
    child2 = _deduplicate_session(ExerciseSession(all_items[mid:]))

    # Ensure children have at least 2 items
    if len(child1) < 2:
        child1 = parent1.copy()
    if len(child2) < 2:
        child2 = parent2.copy()

    return child1, child2


def _deduplicate_session(session: ExerciseSession) -> ExerciseSession:
    """Merges duplicate activities by summing durations."""
    seen: Dict[str, SessionItem] = {}
    for it in session.items:
        if it.activity.name in seen:
            seen[it.activity.name].duration += it.duration
        else:
            seen[it.activity.name] = it.copy()
    return ExerciseSession(list(seen.values()))


# ---------------------------------------------------------
# Mutation Operator
# ---------------------------------------------------------

def standard_mutation(
    session: ExerciseSession,
    activities_pool: Dict[str, Activity] | List[Activity],
    mutation_rate: float = 0.15,
    rng: Optional[random.Random] = None,
) -> ExerciseSession:
    """Mutation: one session in, one session out.

    Subject slides: mutation causes a small random change. Duration is
    shifted by a few minutes, the activity can be replaced, or a gene can
    be inserted or deleted so the length stays in [2, 5]. Same operator
    family as Eiben and Smith (2015), Section 4.5.
    """
    r = rng if rng is not None else random.Random()
    if isinstance(activities_pool, dict):
        pool_list = list(activities_pool.values())
    else:
        pool_list = list(activities_pool)

    mutated = session.copy()

    # 1. Duration perturbation
    for item in mutated.items:
        if r.random() < mutation_rate:
            delta = r.choice([-10.0, -5.0, -2.0, 2.0, 5.0, 10.0])
            item.duration = max(5.0, min(45.0, item.duration + delta))

    # 2. Activity substitution
    if r.random() < mutation_rate and len(mutated.items) > 0:
        idx_to_replace = r.randrange(len(mutated.items))
        current_names = {it.activity.name for it in mutated.items}
        available = [a for a in pool_list if a.name not in current_names]
        if available:
            new_act = r.choice(available)
            mutated.items[idx_to_replace].activity = new_act

    # 3. Activity insertion (if length < 5)
    if r.random() < (mutation_rate * 0.7) and len(mutated.items) < 5:
        current_names = {it.activity.name for it in mutated.items}
        available = [a for a in pool_list if a.name not in current_names]
        if available:
            new_act = r.choice(available)
            dur = round(r.uniform(8.0, 20.0), 0)
            mutated.add_activity(new_act, dur)

    # 4. Activity deletion (if length > 2)
    if r.random() < (mutation_rate * 0.5) and len(mutated.items) > 2:
        del_idx = r.randrange(len(mutated.items))
        mutated.items.pop(del_idx)

    return mutated
