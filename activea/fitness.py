"""Fitness function for ACTIVEA (Tasks 2 and 3).

Lecture baseline (Code 2): +1 inside a target band, -1 outside.
Continuous model, scaled to 100 points:

    F = 30*f_cal + 15*f_MET + 15*f_RPE + 15*f_HR + 25*f_dom - P_dur

    f = max(0, 1 - |x - midpoint| / (high - low))
    f_dom = (domains covered) / 7
    P_dur = 1.5 points per minute outside [20, 75]
    P_struct = 10 when the session has fewer than 2 activities

Calories use ACSM (2018): kcal/min = MET * 3.5 * kg / 200.
RPE is the Borg CR10 scale (Borg, 1998). Heart rate is % of maximum heart rate.
"""

from __future__ import annotations

from .models import ExerciseSession, FitnessResult, FITNESS_DOMAINS


class ActiveaFitnessEvaluator:
    """Evaluates the exercise quality and health optimality of an exercise session
    based on caloric expenditure, MET, RPE, HR percentage, and domain coverage.
    """

    def __init__(
        self,
        goal: str = "standard",  # "standard", "general_health", "fitness_improvement", "weight_loss"
        evaluation_mode: str = "continuous",  # "continuous" or "baseline"
        user_weight_kg: float = 70.0,
        min_duration_min: float = 20.0,
        max_duration_min: float = 75.0,
        weight_cal: float = 30.0,
        weight_met: float = 15.0,
        weight_rpe: float = 15.0,
        weight_hr: float = 15.0,
        weight_domain: float = 25.0,
    ):
        self.goal = goal
        self.evaluation_mode = evaluation_mode
        self.user_weight_kg = user_weight_kg
        self.min_duration_min = min_duration_min
        self.max_duration_min = max_duration_min

        # Weights summing to 100
        self.w_cal = weight_cal
        self.w_met = weight_met
        self.w_rpe = weight_rpe
        self.w_hr = weight_hr
        self.w_domain = weight_domain

        # Configure target ranges based on user goal from specification
        if goal == "general_health":
            self.cal_min, self.cal_max = 200.0, 400.0
            self.target_met_min, self.target_met_max = 3.0, 6.0
            self.target_rpe_min, self.target_rpe_max = 3.0, 6.0
            self.target_hr_min, self.target_hr_max = 55.0, 75.0
        elif goal == "fitness_improvement":
            self.cal_min, self.cal_max = 300.0, 500.0
            self.target_met_min, self.target_met_max = 4.5, 7.5
            self.target_rpe_min, self.target_rpe_max = 4.0, 7.0
            self.target_hr_min, self.target_hr_max = 64.0, 80.0
        elif goal == "weight_loss":
            self.cal_min, self.cal_max = 400.0, 600.0
            self.target_met_min, self.target_met_max = 5.5, 9.0
            self.target_rpe_min, self.target_rpe_max = 5.0, 8.5
            self.target_hr_min, self.target_hr_max = 70.0, 88.0
        else:  # "standard" general recommendation from spec: 300 - 600 calories
            self.cal_min, self.cal_max = 300.0, 600.0
            self.target_met_min, self.target_met_max = 4.0, 8.0
            self.target_rpe_min, self.target_rpe_max = 4.0, 7.5
            self.target_hr_min, self.target_hr_max = 64.0, 82.0

    def evaluate_lecture_baseline(self, session: ExerciseSession) -> int:
        """Faithful baseline implementation of the slide's discrete rule-set from Code 2 (fit.py.txt).
        
        Uses step rewards (+1) and penalties (-1) to demonstrate continuity with taught material.
        """
        fit_val = 0
        tot_cal = session.total_calories(self.user_weight_kg)
        tot_met = session.weighted_met
        tot_rpe = session.weighted_rpe
        tot_hr = session.weighted_hr_pct

        # 1. Caloric condition
        if 300.0 <= tot_cal <= 600.0:
            fit_val += 1
        else:
            fit_val -= 1

        # 2. MET condition
        if tot_met >= 4.0:
            fit_val += 1
        else:
            fit_val -= 1

        # 3. RPE condition
        if 4.0 <= tot_rpe <= 7.0:
            fit_val += 1
        else:
            fit_val -= 1

        # 4. HR condition
        if 64.0 <= tot_hr <= 80.0:
            fit_val += 1
        else:
            fit_val -= 1

        return fit_val

    @staticmethod
    def _band_score(value: float, low: float, high: float) -> float:
        """f = max(0, 1 - |x - midpoint| / (high - low)).

        Peaks at the middle of the prescribed band and is zero outside it.
        """
        span = high - low
        if span <= 0.0:
            return 1.0 if value == low else 0.0
        midpoint = (low + high) / 2.0
        return max(0.0, 1.0 - abs(value - midpoint) / span)

    def evaluate(self, session: ExerciseSession) -> FitnessResult:
        """Evaluates fitness using either continuous multi-attribute scoring or lecture baseline discrete scoring."""
        if self.evaluation_mode == "baseline":
            tot_cal = session.total_calories(self.user_weight_kg)
            tot_met = session.weighted_met
            tot_rpe = session.weighted_rpe
            tot_hr = session.weighted_hr_pct
            base_score = float(self.evaluate_lecture_baseline(session))
            return FitnessResult(
                total_fitness=base_score,
                calorie_score=1.0 if (self.cal_min <= tot_cal <= self.cal_max) else -1.0,
                met_score=1.0 if (tot_met >= self.target_met_min) else -1.0,
                rpe_score=1.0 if (self.target_rpe_min <= tot_rpe <= self.target_rpe_max) else -1.0,
                hr_score=1.0 if (self.target_hr_min <= tot_hr <= self.target_hr_max) else -1.0,
                domain_score=float(len(session.domains_covered)),
                duration_penalty=0.0,
                details={
                    "mode": "lecture_baseline",
                    "total_calories": round(tot_cal, 1),
                    "weighted_met": round(tot_met, 2),
                    "weighted_rpe": round(tot_rpe, 2),
                    "weighted_hr": round(tot_hr, 2),
                    "total_duration": round(session.total_duration, 1),
                    "domains_covered": list(session.domains_covered),
                }
            )

        if len(session) == 0 or session.total_duration <= 0.0:
            return FitnessResult(
                total_fitness=0.0,
                calorie_score=0.0,
                met_score=0.0,
                rpe_score=0.0,
                hr_score=0.0,
                domain_score=0.0,
                duration_penalty=50.0,
                details={"error": "empty session"}
            )

        tot_cal = session.total_calories(self.user_weight_kg)
        weighted_met = session.weighted_met
        weighted_rpe = session.weighted_rpe
        weighted_hr = session.weighted_hr_pct
        tot_dur = session.total_duration
        covered_domains = session.domains_covered

        # Continuous band score used for every intensity component.
        # Eiben and Smith (2015), Ch. 3: a fitness function must give a
        # gradient. The lecture +1/-1 rules are kept in evaluate_lecture_baseline.
        # Component form (coursework model):
        #   f = max(0, 1 - |x - midpoint| / (high - low))
        cal_score = self._band_score(tot_cal, self.cal_min, self.cal_max)
        met_score = self._band_score(weighted_met, self.target_met_min, self.target_met_max)
        rpe_score = self._band_score(weighted_rpe, self.target_rpe_min, self.target_rpe_max)
        hr_score = self._band_score(weighted_hr, self.target_hr_min, self.target_hr_max)

        # 5. Fitness Domain Coverage Score [0.0 - 1.0]
        # Requires broad coverage across all 7 domains: CV, ST, ME, FL, BA, AG, FM
        num_domains = len(covered_domains)
        domain_score = num_domains / float(len(FITNESS_DOMAINS))

        # 6. Duration Constraints & Penalties
        dur_penalty = 0.0
        if tot_dur < self.min_duration_min:
            dur_penalty += (self.min_duration_min - tot_dur) * 1.5
        elif tot_dur > self.max_duration_min:
            dur_penalty += (tot_dur - self.max_duration_min) * 1.5

        # P_struct: one-activity sessions lose 10 points (coursework model).
        if len(session) < 2:
            dur_penalty += 10.0

        # Weighted combination
        weighted_sum = (
            self.w_cal * cal_score +
            self.w_met * met_score +
            self.w_rpe * rpe_score +
            self.w_hr * hr_score +
            self.w_domain * domain_score
        )

        total_fitness = max(0.0, weighted_sum - dur_penalty)

        details = {
            "total_calories": tot_cal,
            "weighted_met": weighted_met,
            "weighted_rpe": weighted_rpe,
            "weighted_hr": weighted_hr,
            "total_duration": tot_dur,
            "domains_covered": list(covered_domains),
            "num_domains": num_domains,
        }

        return FitnessResult(
            total_fitness=round(total_fitness, 3),
            calorie_score=round(cal_score, 3),
            met_score=round(met_score, 3),
            rpe_score=round(rpe_score, 3),
            hr_score=round(hr_score, 3),
            domain_score=round(domain_score, 3),
            duration_penalty=round(dur_penalty, 3),
            preference_penalty=0.0,
            details=details,
        )
