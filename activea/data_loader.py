"""Data setup and dataset loading module for ACTIVEA (Task 1).

Implements:
1. Exercise Quality Knowledge Base completed with literature-verified metrics
   (Ainsworth et al., 2011; ACSM Guidelines, 2018; Borg RPE Scale).
2. Wearable sensor dataset reader (wearables_monitoring_data.csv) with
   train/test separation and empirical physiological profile extraction.
"""

from __future__ import annotations
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .models import Activity


class ExerciseKnowledgeBase:
    """Compiles and manages the pool of supported physical activities and their
    exercise-science intensity metrics as specified in the coursework.
    """

    @classmethod
    def get_default_activities(cls) -> Dict[str, Activity]:
        """Returns the populated exercise quality table matching coursework spec
        requirements and validated against the Ainsworth Compendium of Physical Activities.
        """
        activities: List[Activity] = [
            Activity(
                name="Jogging",
                calories_per_min_70kg=7.35,
                met=6.0,
                target_rpe=5.0,
                target_hr_min=70.0,
                target_hr_max=77.0,
                domains={"CV", "ME", "FM"}
            ),
            Activity(
                name="Bodyweight squats",
                calories_per_min_70kg=4.90,
                met=4.0,
                target_rpe=6.0,
                target_hr_min=64.0,
                target_hr_max=70.0,
                domains={"ST", "ME", "BA", "FM"}
            ),
            Activity(
                name="Cycling (moderate)",
                calories_per_min_70kg=6.80,
                met=5.5,
                target_rpe=4.0,
                target_hr_min=65.0,
                target_hr_max=75.0,
                domains={"CV", "ME"}
            ),
            Activity(
                name="Plank",
                calories_per_min_70kg=3.50,
                met=3.0,
                target_rpe=7.0,
                target_hr_min=60.0,
                target_hr_max=68.0,
                domains={"ST", "ME", "BA"}
            ),
            Activity(
                name="Yoga",
                calories_per_min_70kg=3.00,
                met=2.5,
                target_rpe=3.0,
                target_hr_min=55.0,
                target_hr_max=65.0,
                domains={"FL", "BA", "FM"}
            ),
            Activity(
                name="Rope skipping",
                calories_per_min_70kg=11.00,
                met=9.0,
                target_rpe=8.0,
                target_hr_min=78.0,
                target_hr_max=88.0,
                domains={"CV", "AG", "ME", "FM"}
            ),
            Activity(
                name="Running",
                calories_per_min_70kg=10.50,
                met=8.5,
                target_rpe=7.0,
                target_hr_min=77.0,
                target_hr_max=85.0,
                domains={"CV", "ME", "FM"}
            ),
            Activity(
                name="Brisk walking",
                calories_per_min_70kg=4.10,
                met=3.5,
                target_rpe=3.0,
                target_hr_min=55.0,
                target_hr_max=65.0,
                domains={"CV", "FM"}
            ),
            Activity(
                name="Pushups & Calisthenics",
                calories_per_min_70kg=5.60,
                met=4.5,
                target_rpe=6.0,
                target_hr_min=65.0,
                target_hr_max=75.0,
                domains={"ST", "ME", "FM"}
            ),
            Activity(
                name="Agility drills & Burpees",
                calories_per_min_70kg=9.80,
                met=8.0,
                target_rpe=8.0,
                target_hr_min=80.0,
                target_hr_max=90.0,
                domains={"CV", "AG", "ST", "FM"}
            ),
        ]
        return {act.name: act for act in activities}

    @classmethod
    def to_markdown_table(cls) -> str:
        """Formats the completed exercise quality table into Markdown for the report."""
        domain_order = ["CV", "ST", "ME", "FL", "BA", "AG", "FM"]
        activities = cls.get_default_activities().values()
        header = (
            "| Activity | Calories/min (70kg) | MET | Target RPE (1-10) | "
            "Target HR [% MHR] | Fitness Domains |\n"
            "|---|---|---|---|---|---|\n"
        )
        rows = []
        for a in activities:
            hr_str = f"{a.target_hr_min:.0f}-{a.target_hr_max:.0f}"
            dom_str = ", ".join(sorted(a.domains, key=lambda d: domain_order.index(d) if d in domain_order else 99))
            rows.append(
                f"| {a.name} | {a.calories_per_min_70kg:.2f} | {a.met:.1f} | "
                f"{a.target_rpe:.1f} | {hr_str} | {dom_str} |"
            )
        return header + "\n".join(rows)


class WearableDataLoader:
    """Loads and processes wearables_monitoring_data.csv to calibrate and evaluate
    evolutionary activity recommendations against empirical physiological records.
    """

    def __init__(self, csv_path: str | Path = "exercise_dataset.csv"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            fallback = Path("exercise_dataset.csv")
            if fallback.exists():
                self.csv_path = fallback
        self.records: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        """Parses the CSV file and converts numerical attributes."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.csv_path}")

        records: List[Dict[str, Any]] = []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Support both exercise_dataset.csv and wearables_monitoring_data.csv schemas
                    if "Calories Burn" in row:
                        gender_str = row.get("Gender", "Male").strip().lower()
                        cleaned_row = {
                            "id": int(row.get("ID", 0)),
                            "age": float(row.get("Age", 25.0)),
                            "gender": 1 if gender_str == "male" else 0,
                            "height": 170.0,
                            "weight": float(row.get("Actual Weight", 70.0)),
                            "steps": 0.0,
                            "heart_rate": float(row.get("Heart Rate", 70.0)),
                            "calories": float(row.get("Calories Burn", 0.0)),
                            "distance": 0.0,
                            "resting_heart": 60.0,
                            "intensity_karvonen": float(row.get("Exercise Intensity", 5.0)),
                            "device": "Wearable",
                            "activity": row.get("Exercise", "").strip(),
                        }
                    else:
                        cleaned_row = {
                            "id": int(row.get("Unnamed: 0", 0)),
                            "age": float(row.get("age", 25.0)),
                            "gender": int(row.get("gender", 1)),
                            "height": float(row.get("height", 170.0)),
                            "weight": float(row.get("weight", 70.0)),
                            "steps": float(row.get("steps", 0.0)),
                            "heart_rate": float(row.get("hear_rate", row.get("heart_rate", 70.0))),
                            "calories": float(row.get("calories", 0.0)),
                            "distance": float(row.get("distance", 0.0)),
                            "resting_heart": float(row.get("resting_heart", 60.0)),
                            "intensity_karvonen": float(row.get("intensity_karvonen", 0.0)),
                            "device": row.get("device", ""),
                            "activity": row.get("activity", "").strip(),
                        }
                    records.append(cleaned_row)
                except (ValueError, TypeError):
                    continue
        self.records = records
        return self.records

    def train_test_split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Separates dataset into stratified training and testing partitions by activity.
        
        Satisfies Task 1 requirement: 'Separate training data from testing data'.
        """
        if not self.records:
            self.load_data()

        rng = random.Random(seed)
        # Stratify by activity class
        by_activity: Dict[str, List[Dict[str, Any]]] = {}
        for r in self.records:
            act = r["activity"]
            by_activity.setdefault(act, []).append(r)

        train_set: List[Dict[str, Any]] = []
        test_set: List[Dict[str, Any]] = []

        for act, group in by_activity.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            split_idx = int(len(shuffled) * train_ratio)
            train_set.extend(shuffled[:split_idx])
            test_set.extend(shuffled[split_idx:])

        rng.shuffle(train_set)
        rng.shuffle(test_set)
        return train_set, test_set

    @staticmethod
    def compute_empirical_summary(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Computes statistical profiles (mean & std for calories, HR, intensity) per activity."""
        grouped: Dict[str, Dict[str, List[float]]] = {}
        for r in records:
            act = r["activity"]
            if act not in grouped:
                grouped[act] = {"calories": [], "hr": [], "karvonen": [], "weight": []}
            grouped[act]["calories"].append(r["calories"])
            grouped[act]["hr"].append(r["heart_rate"])
            grouped[act]["karvonen"].append(r["intensity_karvonen"])
            grouped[act]["weight"].append(r["weight"])

        summary: Dict[str, Dict[str, float]] = {}
        for act, metrics in grouped.items():
            n = len(metrics["calories"])
            if n == 0:
                continue

            def mean(lst: List[float]) -> float:
                return sum(lst) / len(lst)

            def std(lst: List[float], m: float) -> float:
                if len(lst) <= 1:
                    return 0.0
                return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

            cal_m = mean(metrics["calories"])
            hr_m = mean(metrics["hr"])
            karv_m = mean(metrics["karvonen"])
            wt_m = mean(metrics["weight"])

            summary[act] = {
                "count": float(n),
                "mean_calories": cal_m,
                "std_calories": std(metrics["calories"], cal_m),
                "mean_hr": hr_m,
                "std_hr": std(metrics["hr"], hr_m),
                "mean_karvonen": karv_m,
                "std_karvonen": std(metrics["karvonen"], karv_m),
                "mean_weight": wt_m,
            }
        return summary
