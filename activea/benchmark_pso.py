"""Particle Swarm Optimization (PSO) benchmark module for ACTIVEA.

Implements continuous Particle Swarm Optimization according to Kennedy & Eberhart
(1995) and the CS3910 Computational Intelligence lecture slides on Particle Swarm
Optimization.

Encodes exercise sessions as D-dimensional continuous duration vectors over the
activity pool, with threshold-activated gene inclusion and velocity clamping.
"""

from __future__ import annotations
import random
import time
from typing import List, Dict, Any, Optional, Tuple

from .models import Activity, ExerciseSession, SessionItem, FitnessResult
from .fitness import ActiveaFitnessEvaluator


class ParticleSwarmOptimizer:
    """Particle Swarm Optimizer operating on continuous activity-duration representations.
    
    Lecture Reference: CS3910 Swarm Intelligence & PSO.
    Particles adjust velocities based on personal best (cognitive attraction)
    and global swarm best (social attraction) with dynamic inertia damping.
    """

    def __init__(
        self,
        activities_pool: Dict[str, Activity] | List[Activity],
        fitness_evaluator: Optional[ActiveaFitnessEvaluator] = None,
        swarm_size: int = 25,
        max_iterations: int = 100,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.496,  # Cognitive attraction coefficient
        c2: float = 1.496,  # Social attraction coefficient
        v_max: float = 10.0,
        inclusion_threshold: float = 5.0,
        seed: Optional[int] = 42,
    ):
        self.activities_pool = (
            list(activities_pool.values()) if isinstance(activities_pool, dict) else list(activities_pool)
        )
        self.evaluator = fitness_evaluator if fitness_evaluator is not None else ActiveaFitnessEvaluator()
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.inclusion_threshold = inclusion_threshold
        self.dim = len(self.activities_pool)
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[ExerciseSession] = None
        self.best_fitness: float = -1.0
        self.best_result: Optional[FitnessResult] = None
        self.execution_time_sec: float = 0.0

    def vector_to_session(self, vec: List[float]) -> ExerciseSession:
        """Decodes continuous D-dimensional vector into a valid ExerciseSession."""
        # Find active components with duration >= threshold
        active = [(idx, val) for idx, val in enumerate(vec) if val >= self.inclusion_threshold]

        # Enforce [2, 5] activity constraint
        if len(active) < 2:
            # Pick top 2 coordinates with largest values
            ranked = sorted(enumerate(vec), key=lambda x: x[1], reverse=True)
            active = [(ranked[0][0], max(10.0, ranked[0][1])), (ranked[1][0], max(10.0, ranked[1][1]))]
        elif len(active) > 5:
            # Retain top 5 highest durations
            active = sorted(active, key=lambda x: x[1], reverse=True)[:5]

        session = ExerciseSession()
        for idx, raw_dur in active:
            clamped_dur = round(max(5.0, min(45.0, raw_dur)), 1)
            session.add_activity(self.activities_pool[idx], clamped_dur)
        return session

    def session_to_vector(self, session: ExerciseSession) -> List[float]:
        """Encodes an ExerciseSession into a D-dimensional vector."""
        vec = [0.0] * self.dim
        for item in session.items:
            for idx, act in enumerate(self.activities_pool):
                if act.name == item.activity.name:
                    vec[idx] = item.duration
                    break
        return vec

    def optimize(self) -> Tuple[ExerciseSession, FitnessResult, List[Dict[str, Any]]]:
        """Runs Particle Swarm Optimization."""
        start_time = time.time()
        self.history = []

        # 1. Swarm Initialisation
        positions: List[List[float]] = []
        velocities: List[List[float]] = []
        pbest_positions: List[List[float]] = []
        pbest_fitnesses: List[float] = []
        pbest_sessions: List[ExerciseSession] = []
        pbest_results: List[FitnessResult] = []

        gbest_position: Optional[List[float]] = None
        gbest_fitness: float = -1.0
        gbest_session: Optional[ExerciseSession] = None
        gbest_result: Optional[FitnessResult] = None

        eval_count = 0

        for _ in range(self.swarm_size):
            # Generate random position: 2 to 4 activities selected
            pos = [0.0] * self.dim
            num_chosen = self.rng.randint(2, 4)
            chosen_indices = self.rng.sample(range(self.dim), num_chosen)
            for idx in chosen_indices:
                pos[idx] = self.rng.uniform(10.0, 30.0)

            # Initial velocity in [-v_max/2, v_max/2]
            vel = [self.rng.uniform(-self.v_max / 2.0, self.v_max / 2.0) for _ in range(self.dim)]

            session = self.vector_to_session(pos)
            res = self.evaluator.evaluate(session)
            fit = res.total_fitness
            eval_count += 1

            positions.append(pos)
            velocities.append(vel)
            pbest_positions.append(list(pos))
            pbest_fitnesses.append(fit)
            pbest_sessions.append(session)
            pbest_results.append(res)

            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_position = list(pos)
                gbest_session = session.copy()
                gbest_result = res

        self.best_individual = gbest_session.copy()
        self.best_fitness = gbest_fitness
        self.best_result = gbest_result

        mean_fit = sum(pbest_fitnesses) / len(pbest_fitnesses)
        self.history.append({
            "step": 0,
            "iteration": 0,
            "evaluation": eval_count,
            "best_fitness": round(self.best_fitness, 3),
            "mean_fitness": round(mean_fit, 3),
            "best_session": self.best_individual.to_tuple_list(),
            "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
            "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
        })

        # 2. Main PSO Iteration Loop
        for iteration in range(1, self.max_iterations + 1):
            # Time-varying inertia weight
            w = self.w_start - ((self.w_start - self.w_end) * (iteration / self.max_iterations))

            iter_fitnesses: List[float] = []

            for i in range(self.swarm_size):
                # Update velocity & position per dimension
                for d in range(self.dim):
                    r1 = self.rng.random()
                    r2 = self.rng.random()
                    cog = self.c1 * r1 * (pbest_positions[i][d] - positions[i][d])
                    soc = self.c2 * r2 * (gbest_position[d] - positions[i][d])
                    new_v = w * velocities[i][d] + cog + soc
                    # Clamp velocity
                    velocities[i][d] = max(-self.v_max, min(self.v_max, new_v))
                    # Update position
                    positions[i][d] = max(0.0, min(45.0, positions[i][d] + velocities[i][d]))

                # Evaluate candidate
                sess = self.vector_to_session(positions[i])
                res = self.evaluator.evaluate(sess)
                fit = res.total_fitness
                eval_count += 1
                iter_fitnesses.append(fit)

                # Update personal best
                if fit > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = list(positions[i])
                    pbest_sessions[i] = sess
                    pbest_results[i] = res

                    # Update global best
                    if fit > gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = list(positions[i])
                        gbest_session = sess.copy()
                        gbest_result = res
                        self.best_fitness = fit
                        self.best_individual = sess.copy()
                        self.best_result = res

            mean_fit = sum(iter_fitnesses) / len(iter_fitnesses)
            self.history.append({
                "step": iteration,
                "iteration": iteration,
                "evaluation": eval_count,
                "best_fitness": round(self.best_fitness, 3),
                "mean_fitness": round(mean_fit, 3),
                "best_session": self.best_individual.to_tuple_list(),
                "best_calories": round(self.best_result.details.get("total_calories", 0.0), 1),
                "best_duration": round(self.best_result.details.get("total_duration", 0.0), 1),
            })

        self.execution_time_sec = round(time.time() - start_time, 4)
        return self.best_individual, self.best_result, self.history
