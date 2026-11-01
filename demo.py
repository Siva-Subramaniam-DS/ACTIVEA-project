"""Interactive Live Demonstration Script for ACTIVEA Presentation.

Designed for the 8-minute coursework presentation and live demo.
Runs live evolutionary optimization to generate personalized workout plans.
"""

import sys
import time
from activea.models import ExerciseSession
from activea.data_loader import ExerciseKnowledgeBase
from activea.fitness import ActiveaFitnessEvaluator
from activea.core_ga import ActiveaGeneticAlgorithm
from activea.advanced import DomainPreferenceFitnessEvaluator


def main():
    print("=" * 70)
    print("  ACTIVEA: Physical Activity Planning Evolutionary Algorithm")
    print("  Live Software Demonstration (Aston University - DG4COM)")
    print("=" * 70)

    kb = ExerciseKnowledgeBase.get_default_activities()
    print(f"\n[System Loaded] Candidate activity pool initialized ({len(kb)} activities).")

    print("\nSelect Training Profile for Live Evolution:")
    print("  1. Balanced General Health (300-600 kcal, Holistic 7-Domain)")
    print("  2. Functional Cardiovascular Endurance (Focus: CV & FM)")
    print("  3. Muscular Strength & Conditioning (Focus: ST & ME)")

    choice = "1"
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        try:
            user_input = input("\nEnter choice [1-3] (Default: 1): ").strip()
            if user_input in ["1", "2", "3"]:
                choice = user_input
        except EOFError:
            choice = "1"

    if choice == "2":
        profile_name = "Functional Cardiovascular Endurance (CV + FM)"
        evaluator = DomainPreferenceFitnessEvaluator(
            preferred_domains={"CV", "FM"},
            preference_penalty_weight=2.5,
            preference_bonus_weight=15.0,
            goal="standard"
        )
    elif choice == "3":
        profile_name = "Muscular Strength & Conditioning (ST + ME)"
        evaluator = DomainPreferenceFitnessEvaluator(
            preferred_domains={"ST", "ME"},
            preference_penalty_weight=2.5,
            preference_bonus_weight=15.0,
            goal="standard"
        )
    else:
        profile_name = "Balanced General Health (Holistic 7-Domain)"
        evaluator = ActiveaFitnessEvaluator(goal="standard", user_weight_kg=70.0)

    print(f"\n[Target Selected] -> {profile_name}")
    print("[Optimizer Starting] Running ActiveaGeneticAlgorithm (Pop=50, Gens=40)...")

    start = time.time()
    ga = ActiveaGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=evaluator,
        pop_size=50,
        max_generations=40,
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=3,
        seed=int(time.time()) % 1000,
    )

    best_session, best_result, history = ga.evolve()
    elapsed = time.time() - start

    print(f"[Evolution Complete] Optimization converged in {elapsed:.3f} seconds.")
    print("-" * 70)
    print(f"  Total Fitness Score : {best_result.total_fitness:.2f} / 100.0")
    print(f"  Total Workout Time  : {best_session.total_duration:.1f} minutes")
    print(f"  ACSM Energy (Target): {best_session.total_calories(70.0):.1f} kcal (Clinical Target: 300-600 kcal)")

    # ML Champion Model Predicted Energy
    try:
        from evaluate_with_model import MLSavedModelFitnessEvaluator
        ml_eval = MLSavedModelFitnessEvaluator(goal="standard", user_weight_kg=70.0, age=30.0)
        session_tuples = [(it.activity.name, it.duration) for it in best_session.items]
        ml_calories = ml_eval.predict_calories(session_tuples)
        print(f"  ML Predicted Energy : {ml_calories:.1f} kcal (Champion Gradient Boosting Model)")
    except Exception:
        pass

    print(f"  Average Intensity   : MET {best_session.weighted_met:.2f} | RPE {best_session.weighted_rpe:.2f} (Borg 1-10)")
    print(f"  Target Heart Rate   : {best_session.weighted_hr_pct:.1f}% MHR")
    print(f"  Fitness Domains     : {', '.join(sorted(best_session.domains_covered))}")
    print("-" * 70)
    print("\nRecommended Session Workout Routine:")
    print(f"  {'#':<3} {'Activity':<26} {'Duration':<12} {'Calories':<12} {'Domains Covered'}")
    print(f"  {'-'*3} {'-'*26} {'-'*12} {'-'*12} {'-'*20}")
    for idx, item in enumerate(best_session.items, 1):
        cal = item.activity.calories_per_min(70.0) * item.duration
        doms = ", ".join(sorted(item.activity.domains))
        print(f"  {idx:<3} {item.activity.name:<26} {item.duration:.1f} min{'':<5} {cal:.1f} kcal{'':<4} {doms}")

    print("\n" + "=" * 70)
    print("  Ready for Q&A discussion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
