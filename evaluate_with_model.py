"""ACTIVEA - Machine Learning Surrogate Fitness Evaluator
Loads the trained Champion Gradient Boosting Model ('Model/best_calorie_model.joblib')
and evaluates candidate exercise sessions.

Use this evaluator in any evolutionary or metaheuristic optimization algorithm:
    from evaluate_with_model import MLSavedModelFitnessEvaluator, EXERCISE_POOL

    evaluator = MLSavedModelFitnessEvaluator(goal="fitness_improvement", user_weight_kg=75.0, age=26)
    candidate_session = [("Jogging", 15.0), ("Bodyweight squats", 12.0)]
    fitness, details = evaluator.evaluate(candidate_session)
"""

import sys
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Set

import numpy as np
import pandas as pd
import joblib

# Cross-version scikit-learn unpickling compatibility
warnings.filterwarnings("ignore")
try:
    import sklearn.compose._column_transformer as column_transformer
    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

try:
    from sklearn._loss import _loss
    sys.modules["_loss"] = _loss
except Exception:
    pass


@dataclass
class ActivityDef:
    name: str
    sport_modality: str      # Matches endomondoHR.csv sport category
    met: float
    target_rpe: float
    target_hr_pct: float
    domains: Set[str] = field(default_factory=set)
    base_speed_kmh: float = 8.0


# Coursework Standard Activities mapped to Endomondo sports telemetry
EXERCISE_POOL: Dict[str, ActivityDef] = {
    "Jogging": ActivityDef("Jogging", "run", 6.0, 5.0, 73.5, {"CV", "ME", "FM"}, 8.5),
    "Bodyweight squats": ActivityDef("Bodyweight squats", "core stability training", 4.0, 6.0, 67.0, {"ST", "ME", "BA", "FM"}, 2.0),
    "Cycling (moderate)": ActivityDef("Cycling (moderate)", "bike", 5.5, 4.0, 70.0, {"CV", "ME"}, 22.0),
    "Plank": ActivityDef("Plank", "core stability training", 3.0, 7.0, 64.0, {"ST", "ME", "BA"}, 1.0),
    "Yoga": ActivityDef("Yoga", "core stability training", 2.5, 3.0, 60.0, {"FL", "BA", "FM"}, 1.0),
    "Rope skipping": ActivityDef("Rope skipping", "indoor cycling", 9.0, 8.0, 83.0, {"CV", "AG", "ME", "FM"}, 12.0),
    "Running": ActivityDef("Running", "run", 8.5, 7.0, 81.0, {"CV", "ME", "FM"}, 10.5),
    "Brisk walking": ActivityDef("Brisk walking", "walk", 3.5, 3.0, 60.0, {"CV", "FM"}, 5.5),
    "Pushups & Calisthenics": ActivityDef("Pushups & Calisthenics", "core stability training", 4.5, 6.0, 70.0, {"ST", "ME", "FM"}, 2.5),
    "Agility drills & Burpees": ActivityDef("Agility drills & Burpees", "orienteering", 8.0, 8.0, 85.0, {"CV", "AG", "ST", "FM"}, 9.0),
}


class MLSavedModelFitnessEvaluator:
    """Evaluates candidate workout sessions using the trained ML model for calories
    combined with the ACSM/Borg multi-attribute continuous fitness function.
    """

    GOALS = {
        "general_health": {"cal": (200.0, 400.0), "met": (3.0, 6.0), "rpe": (3.0, 6.0), "hr": (55.0, 75.0)},
        "fitness_improvement": {"cal": (300.0, 500.0), "met": (4.5, 7.5), "rpe": (4.0, 7.0), "hr": (64.0, 80.0)},
        "weight_loss": {"cal": (400.0, 600.0), "met": (5.5, 9.0), "rpe": (5.0, 8.5), "hr": (70.0, 88.0)},
        "standard": {"cal": (300.0, 600.0), "met": (4.0, 8.0), "rpe": (4.0, 7.5), "hr": (64.0, 82.0)}
    }

    def __init__(
        self,
        model_path: str = "Model/best_calorie_model.joblib",
        goal: str = "fitness_improvement",
        user_weight_kg: float = 75.0,
        age: float = 26.0,
        gender: str = "male",
        min_duration_min: float = 20.0,
        max_duration_min: float = 75.0
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.goal = goal
        self.user_weight_kg = float(user_weight_kg)
        self.age = float(age)
        self.gender = gender.lower()
        self.min_duration_min = float(min_duration_min)
        self.max_duration_min = float(max_duration_min)

        band = self.GOALS.get(goal, self.GOALS["fitness_improvement"])
        self.cal_range = band["cal"]
        self.met_range = band["met"]
        self.rpe_range = band["rpe"]
        self.hr_range = band["hr"]

    @staticmethod
    def _band_score(value: float, low: float, high: float) -> float:
        span = high - low
        if span <= 0:
            return 1.0 if value == low else 0.0
        midpoint = (low + high) / 2.0
        return max(0.0, 1.0 - abs(value - midpoint) / span)

    def predict_calories(self, session_items: List[Tuple[str, float]]) -> float:
        """Uses the ML model to predict calories for the session."""
        if not session_items:
            return 0.0
        rows = []
        for name, dur in session_items:
            act = EXERCISE_POOL.get(name)
            if not act:
                continue
            hr_bpm = 60.0 + (act.target_hr_pct / 100.0) * (220.0 - self.age - 60.0)
            dist_km = act.base_speed_kmh * (dur / 60.0)
            rows.append({
                "duration_min": float(dur),
                "heart_rate_avg": hr_bpm,
                "heart_rate_max": hr_bpm + 15.0,
                "heart_rate_std": 8.5,
                "speed_avg_kmh": act.base_speed_kmh,
                "elevation_gain_m": 15.0,
                "distance_km": dist_km,
                "sport": act.sport_modality,
                "gender": self.gender
            })
        if not rows:
            return 0.0
        preds = self.model.predict(pd.DataFrame(rows))
        return float(np.sum(preds))

    def evaluate(self, session_items: List[Tuple[str, float]]) -> Tuple[float, Dict[str, Any]]:
        """Evaluates a session and returns (fitness_score [0-100], metrics_dict).
        
        Session format: [("ActivityName", duration_minutes), ...]
        Example: [("Jogging", 15.0), ("Bodyweight squats", 10.0)]
        """
        if not session_items:
            return 0.0, {"error": "empty session"}

        valid_items = [(name, float(dur)) for name, dur in session_items if name in EXERCISE_POOL and dur > 0]
        if not valid_items:
            return 0.0, {"error": "no valid activities"}

        total_duration = sum(dur for _, dur in valid_items)
        if total_duration <= 0.0:
            return 0.0, {"error": "zero duration"}

        # 1. ML-Predicted Caloric Expenditure
        pred_calories = self.predict_calories(valid_items)

        # 2. Exercise-Science Physiological Metrics
        weighted_met = sum(EXERCISE_POOL[name].met * dur for name, dur in valid_items) / total_duration
        weighted_rpe = sum(EXERCISE_POOL[name].target_rpe * dur for name, dur in valid_items) / total_duration
        weighted_hr = sum(EXERCISE_POOL[name].target_hr_pct * dur for name, dur in valid_items) / total_duration
        covered_domains = set().union(*[EXERCISE_POOL[name].domains for name, _ in valid_items])

        # 3. Band Fitness Scores [0.0 - 1.0]
        f_cal = self._band_score(pred_calories, *self.cal_range)
        f_met = self._band_score(weighted_met, *self.met_range)
        f_rpe = self._band_score(weighted_rpe, *self.rpe_range)
        f_hr = self._band_score(weighted_hr, *self.hr_range)
        f_dom = len(covered_domains) / 7.0

        # 4. Duration and Structural Penalties
        p_dur = 0.0
        if total_duration < self.min_duration_min:
            p_dur += (self.min_duration_min - total_duration) * 1.5
        elif total_duration > self.max_duration_min:
            p_dur += (total_duration - self.max_duration_min) * 1.5
        if len(valid_items) < 2:
            p_dur += 10.0  # Encourage multi-component exercise

        # 5. Composite Fitness Function (Scale 0 - 100)
        # F = 30*f_cal + 15*f_MET + 15*f_RPE + 15*f_HR + 25*f_dom - P_dur
        fitness = max(0.0, min(100.0, 30.0 * f_cal + 15.0 * f_met + 15.0 * f_rpe + 15.0 * f_hr + 25.0 * f_dom - p_dur))

        details = {
            "fitness": round(fitness, 2),
            "predicted_calories": round(pred_calories, 1),
            "target_calorie_range": self.cal_range,
            "total_duration_min": round(total_duration, 1),
            "weighted_met": round(weighted_met, 2),
            "weighted_rpe": round(weighted_rpe, 2),
            "weighted_hr_pct": round(weighted_hr, 1),
            "domains_covered": sorted(list(covered_domains)),
            "num_domains": len(covered_domains),
            "session_items": valid_items
        }
        return round(fitness, 2), details


if __name__ == "__main__":
    print("Testing ML Saved Model Fitness Evaluator...")
    evaluator = MLSavedModelFitnessEvaluator(goal="fitness_improvement")
    test_session = [("Jogging", 15.0), ("Bodyweight squats", 12.0)]
    score, info = evaluator.evaluate(test_session)
    print(f"Candidate Session : {test_session}")
    print(f"Fitness Score     : {score} / 100")
    print(f"Predicted Calories: {info['predicted_calories']} kcal (Target: {info['target_calorie_range']})")
    print(f"Weighted MET      : {info['weighted_met']}")
    print(f"Covered Domains   : {info['domains_covered']}")
    print("Ready for your remaining algorithms!")

