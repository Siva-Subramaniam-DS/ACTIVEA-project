"""Data models for ACTIVEA (Physical Activity Planning Evolutionary Algorithm).

Defines the representation of individual activities, structured workout sessions,
and comprehensive fitness evaluation records.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Tuple


# Valid fitness domains defined in the coursework specification
FITNESS_DOMAINS: Set[str] = {
    "CV",  # Cardiovascular Fitness
    "ST",  # Strength Development
    "ME",  # Muscular Endurance
    "FL",  # Flexibility / Mobility
    "BA",  # Balance & Stability
    "AG",  # Agility
    "FM",  # Functional Movement
}


@dataclass(frozen=True)
class Activity:
    """Represents a discrete physical exercise activity with physiological metrics."""
    name: str
    calories_per_min_70kg: float
    met: float
    target_rpe: float
    target_hr_min: float  # Percentage of Maximum Heart Rate (% MHR, e.g. 70.0)
    target_hr_max: float  # Percentage of Maximum Heart Rate (% MHR, e.g. 77.0)
    domains: Set[str] = field(default_factory=set)

    def calories_per_min(self, user_weight_kg: float = 70.0) -> float:
        """Energy expenditure in kcal/min.

        The coursework table gives kcal/min for a 70 kg adult. The brief says
        those rates are modified for the person's weight:
            kcal/min = (kcal/min at 70 kg) * (body mass / 70)
        """
        return self.calories_per_min_70kg * (float(user_weight_kg) / 70.0)

    @property
    def target_hr_mid(self) -> float:
        """Returns the midpoint of the target heart rate percentage range."""
        return (self.target_hr_min + self.target_hr_max) / 2.0


@dataclass
class SessionItem:
    """Represents a specific activity scheduled for a given duration in minutes."""
    activity: Activity
    duration: float  # In minutes

    def copy(self) -> SessionItem:
        return SessionItem(activity=self.activity, duration=float(self.duration))


class ExerciseSession:
    """Represents a complete workout session (an individual chromosome in ACTIVEA).
    
    Expressed as a collection of activities with associated durations:
    session = [(activity_1, duration_1), (activity_2, duration_2), ...]
    """

    def __init__(self, items: List[SessionItem] | None = None):
        self.items: List[SessionItem] = items if items is not None else []

    def add_activity(self, activity: Activity, duration: float) -> None:
        """Appends an activity item to this session."""
        self.items.append(SessionItem(activity=activity, duration=float(duration)))

    @property
    def total_duration(self) -> float:
        """Returns total duration of the workout session in minutes."""
        return sum(item.duration for item in self.items)

    def total_calories(self, user_weight_kg: float = 70.0) -> float:
        """Session energy: sum of table kcal/min, scaled by body mass / 70, times minutes."""
        return sum(
            item.activity.calories_per_min(user_weight_kg) * item.duration
            for item in self.items
        )

    @property
    def weighted_met(self) -> float:
        """Duration-weighted mean MET. Session intensity is the time-weighted mean, not the sum of rates."""
        tot_dur = self.total_duration
        if tot_dur <= 0:
            return 0.0
        return sum(item.activity.met * item.duration for item in self.items) / tot_dur

    @property
    def weighted_rpe(self) -> float:
        """Calculates duration-weighted average Rate of Perceived Exertion (RPE)."""
        tot_dur = self.total_duration
        if tot_dur <= 0:
            return 0.0
        return sum(item.activity.target_rpe * item.duration for item in self.items) / tot_dur

    @property
    def weighted_hr_pct(self) -> float:
        """Calculates duration-weighted average Target Heart Rate percentage (% MHR)."""
        tot_dur = self.total_duration
        if tot_dur <= 0:
            return 0.0
        return sum(item.activity.target_hr_mid * item.duration for item in self.items) / tot_dur

    @property
    def domains_covered(self) -> Set[str]:
        """Returns the set of distinct fitness domains stimulated during this session."""
        covered: Set[str] = set()
        for item in self.items:
            covered.update(item.activity.domains)
        return covered

    def domain_duration_breakdown(self) -> Dict[str, float]:
        """Returns the total active minutes contributing to each fitness domain."""
        breakdown: Dict[str, float] = {d: 0.0 for d in FITNESS_DOMAINS}
        for item in self.items:
            for d in item.activity.domains:
                if d in breakdown:
                    breakdown[d] += item.duration
        return breakdown

    def copy(self) -> ExerciseSession:
        """Creates a deep copy of this exercise session."""
        return ExerciseSession([item.copy() for item in self.items])

    def to_tuple_list(self) -> List[Tuple[str, float]]:
        """Returns simplified tuple representation [(name, duration), ...]."""
        return [(item.activity.name, round(item.duration, 1)) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        tuples_str = ", ".join(f"({it.activity.name}, {it.duration:.1f}m)" for it in self.items)
        return f"ExerciseSession([{tuples_str}], Dur={self.total_duration:.1f}m)"


@dataclass
class FitnessResult:
    """Detailed breakdown of fitness evaluation metrics for transparency and reporting."""
    total_fitness: float
    calorie_score: float
    met_score: float
    rpe_score: float
    hr_score: float
    domain_score: float
    duration_penalty: float
    preference_penalty: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
