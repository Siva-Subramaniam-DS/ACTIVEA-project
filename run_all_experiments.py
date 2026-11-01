"""Master experimental execution script for ACTIVEA coursework.

Runs all tasks (1 through 8), logs numerical results, generates publication-ready
SVG figures, and writes Markdown tables for direct inclusion in the academic report.
"""

from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path

from activea.models import Activity, ExerciseSession, FITNESS_DOMAINS
from activea.data_loader import ExerciseKnowledgeBase, WearableDataLoader
from activea.fitness import ActiveaFitnessEvaluator
from activea.core_ga import ActiveaGeneticAlgorithm
from activea.benchmark_sa import SimulatedAnnealingOptimizer
from activea.experiments import CrowdingGeneticAlgorithm, ParameterTuningExperiment
from activea.advanced import BespokeRepairOperator, DomainPreferenceFitnessEvaluator
from activea.visualizer import SvgChartGenerator


def run_full_coursework_pipeline():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    figures_dir = Path("results/figures")
    figures_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ACTIVEA: Physical Activity Planning Evolutionary Algorithm")
    print("Computational Intelligence Coursework Pipeline Execution")
    print("=" * 70)

    # -------------------------------------------------------------
    # TASK 1: Data Setup & Preprocessing
    # -------------------------------------------------------------
    print("\n[TASK 1] Compiling Exercise Quality Table & Loading Wearable Data...")
    kb = ExerciseKnowledgeBase.get_default_activities()
    kb_md = ExerciseKnowledgeBase.to_markdown_table()
    with open(output_dir / "task1_exercise_quality_table.md", "w", encoding="utf-8") as f:
        f.write("# Exercise Quality Knowledge Base (Task 1)\n\n" + kb_md)
    print("  -> Exercise Quality Knowledge Base saved.")

    wearables_file = Path("exercise_dataset.csv") if Path("exercise_dataset.csv").exists() else Path("wearables_monitoring_data.csv")
    loader = WearableDataLoader(wearables_file)
    all_records = loader.load_data()
    train_data, test_data = loader.train_test_split(train_ratio=0.8, seed=42)
    empirical_train = loader.compute_empirical_summary(train_data)
    empirical_test = loader.compute_empirical_summary(test_data)

    print(f"  -> Total wearable records loaded: {len(all_records)}")
    print(f"  -> Training records: {len(train_data)} | Testing records: {len(test_data)}")

    # Format empirical summary table
    summary_md = [
        "| Activity | Count (Train) | Mean Calories | Mean HR (bpm) | Mean Karvonen Intensity |",
        "|---|---|---|---|---|",
    ]
    for act, s in sorted(empirical_train.items()):
        summary_md.append(
            f"| {act} | {int(s['count'])} | {s['mean_calories']:.2f} ± {s['std_calories']:.2f} | "
            f"{s['mean_hr']:.1f} ± {s['std_hr']:.1f} | {s['mean_karvonen']:.3f} |"
        )
    with open(output_dir / "task1_wearable_summary.md", "w", encoding="utf-8") as f:
        f.write("# Empirical Wearable Dataset Summary (Task 1)\n\n" + "\n".join(summary_md))

    # -------------------------------------------------------------
    # TASK 2 & 3: Core ACTIVEA Algorithm Execution
    # -------------------------------------------------------------
    print("\n[TASK 2 & 3] Running Core ACTIVEA Evolutionary Search...")
    evaluator_standard = ActiveaFitnessEvaluator(goal="standard", user_weight_kg=70.0)

    core_ga = ActiveaGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=evaluator_standard,
        pop_size=50,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=3,
        elitism_count=2,
        seed=42,
    )

    best_session, best_result, ga_history = core_ga.evolve()

    print(f"  -> Best Evolved Fitness: {best_result.total_fitness:.2f} / 100.0")
    print(f"  -> Best Workout Duration: {best_session.total_duration:.1f} min")
    print(f"  -> Caloric Expenditure: {best_session.total_calories(70.0):.1f} kcal")
    print(f"  -> Weighted MET: {best_session.weighted_met:.2f} | Weighted RPE: {best_session.weighted_rpe:.2f}")
    print(f"  -> Weighted HR%: {best_session.weighted_hr_pct:.1f}% MHR")
    print(f"  -> Domains Covered ({len(best_session.domains_covered)}/7): {sorted(best_session.domains_covered)}")
    print(f"  -> Execution Time: {core_ga.execution_time_sec:.3f}s")
    print("  -> Session Composition:")
    for item in best_session.items:
        print(f"     * {item.activity.name}: {item.duration:.1f} min ({', '.join(sorted(item.activity.domains))})")

    # Save core session to markdown
    core_md = [
        f"# Core ACTIVEA Best Workout Session (Task 3)\n",
        f"- **Total Fitness**: {best_result.total_fitness:.2f} / 100",
        f"- **Duration**: {best_session.total_duration:.1f} minutes",
        f"- **Calories Burned (70kg)**: {best_session.total_calories(70.0):.1f} kcal",
        f"- **Duration-weighted MET**: {best_session.weighted_met:.2f}",
        f"- **Duration-weighted RPE**: {best_session.weighted_rpe:.2f} (Borg 1-10 scale)",
        f"- **Target Heart Rate**: {best_session.weighted_hr_pct:.1f}% MHR",
        f"- **Domains Covered**: {', '.join(sorted(best_session.domains_covered))}",
        f"- **Execution Time**: {core_ga.execution_time_sec:.3f} s\n",
        "### Activity Breakdown",
        "| Activity | Duration (min) | Cal/min | Calories | MET | Target RPE | Domains |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in best_session.items:
        cals = it.activity.calories_per_min(70.0) * it.duration
        doms = ", ".join(sorted(it.activity.domains))
        core_md.append(
            f"| {it.activity.name} | {it.duration:.1f} | {it.activity.calories_per_min(70.0):.2f} | "
            f"{cals:.1f} | {it.activity.met:.1f} | {it.activity.target_rpe:.1f} | {doms} |"
        )
    with open(output_dir / "task3_core_session_result.md", "w", encoding="utf-8") as f:
        f.write("\n".join(core_md))

    # -------------------------------------------------------------
    # TASK 4: Nature-Inspired Optimization (Simulated Annealing)
    # -------------------------------------------------------------
    print("\n[TASK 4] Running Nature-Inspired Benchmark (Simulated Annealing)...")
    sa_optimizer = SimulatedAnnealingOptimizer(
        activities_pool=kb,
        fitness_evaluator=evaluator_standard,
        total_evaluations=50 * 50,  # Exactly matching GA evaluations budget
        initial_temperature=100.0,
        cooling_rate=0.996,
        seed=42,
    )
    sa_best_session, sa_best_result, sa_history = sa_optimizer.optimize()

    print(f"  -> SA Best Fitness: {sa_best_result.total_fitness:.2f} / 100.0")
    print(f"  -> SA Total Duration: {sa_best_session.total_duration:.1f} min | Calories: {sa_best_session.total_calories(70.0):.1f} kcal")
    print(f"  -> SA Execution Time: {sa_optimizer.execution_time_sec:.3f}s")

    # Generate Comparison Plot: ACTIVEA vs SA
    ga_curve = [h["best_fitness"] for h in ga_history]
    sa_curve = [h["best_fitness"] for h in sa_history][:len(ga_curve)]
    # Pad sa_curve if needed
    while len(sa_curve) < len(ga_curve):
        sa_curve.append(sa_curve[-1])

    fig1_svg = SvgChartGenerator.generate_convergence_plot(
        series_data={
            "ACTIVEA (Genetic Algorithm)": ga_curve,
            "Simulated Annealing (SA)": sa_curve,
        },
        title="Task 4: Convergence Trajectory - ACTIVEA (GA) vs Simulated Annealing (SA)",
        x_label="Generation / Sampling Interval",
        y_label="Fitness Score (0-100)",
    )
    SvgChartGenerator.save_svg(fig1_svg, figures_dir / "fig1_convergence_ga_vs_sa.svg")
    print("  -> Fig 1 (ACTIVEA vs SA Convergence) saved.")

    # -------------------------------------------------------------
    # TASK 5: Systematic Parameter Management
    # -------------------------------------------------------------
    print("\n[TASK 5] Running Parameter Sensitivity Grid Experiments...")
    tuning_exp = ParameterTuningExperiment(activities_pool=kb)
    param_results = tuning_exp.run_grid_experiment(
        pop_sizes=[20, 50, 100],
        mutation_rates=[0.05, 0.15, 0.30],
        crossover_rates=[0.6, 0.8, 0.95],
        tournament_sizes=[2, 3, 5],
        num_seeds=5,
        generations=35,
    )

    param_md = [
        "# Parameter Sensitivity Analysis (Task 5)\n",
        "| Configuration | Pop Size | Mutation Rate | Crossover Rate | Tourn Size | Mean Best Fitness | Std Dev | Exec Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in param_results:
        param_md.append(
            f"| {r['label']} | {r['pop_size']} | {r['mutation_rate']:.2f} | {r['crossover_rate']:.2f} | "
            f"{r['tourn_size']} | {r['mean_best_fitness']:.2f} | {r['std_best_fitness']:.2f} | {r['mean_exec_time_sec']:.3f} |"
        )
    with open(output_dir / "task5_parameter_tuning.md", "w", encoding="utf-8") as f:
        f.write("\n".join(param_md))
    print("  -> Task 5 Parameter Results saved.")

    # -------------------------------------------------------------
    # TASK 6: Diversity Analysis (Standard GA vs Deterministic Crowding)
    # -------------------------------------------------------------
    print("\n[TASK 6] Evaluating Population Diversity Preservation (Deterministic Crowding)...")
    crowding_ga = CrowdingGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=evaluator_standard,
        pop_size=50,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.15,
        seed=42,
    )
    crowd_best, crowd_res, crowd_history = crowding_ga.evolve()

    std_div = [h["diversity"] for h in ga_history]
    crowd_div = [h["diversity"] for h in crowd_history]

    fig3_svg = SvgChartGenerator.generate_convergence_plot(
        series_data={
            "Deterministic Crowding GA": crowd_div,
            "Standard ACTIVEA GA": std_div,
        },
        title="Task 6: Population Diversity Trajectory Across Generations",
        x_label="Generation",
        y_label="Phenotypic Diversity (Euclidean Gene Distance)",
    )
    SvgChartGenerator.save_svg(fig3_svg, figures_dir / "fig3_diversity_preservation.svg")
    print(f"  -> Final Diversity - Standard GA: {std_div[-1]:.2f} | Crowding GA: {crowd_div[-1]:.2f}")
    print("  -> Fig 3 (Diversity Preservation) saved.")

    # -------------------------------------------------------------
    # TASK 7: Advanced Bespoke Repair Genetic Operator
    # -------------------------------------------------------------
    print("\n[TASK 7] Evaluating Advanced Bespoke Domain-Guided Repair Operator...")
    repair_operator = BespokeRepairOperator(activities_pool=kb)

    # Run GA augmented with bespoke repair operator
    bespoke_ga = ActiveaGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=evaluator_standard,
        pop_size=50,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.15,
        repair_operator=repair_operator,
        seed=42,
    )
    bespoke_best, bespoke_res, bespoke_history = bespoke_ga.evolve()

    bespoke_fits = [h["best_fitness"] for h in bespoke_history]
    std_fits = [h["best_fitness"] for h in ga_history]

    fig4_svg = SvgChartGenerator.generate_convergence_plot(
        series_data={
            "Bespoke Repair GA (Task 7)": bespoke_fits,
            "Standard ACTIVEA GA": std_fits,
        },
        title="Task 7: Convergence Speedup with Bespoke Caloric & Domain Repair Operator",
        x_label="Generation",
        y_label="Fitness Score (0-100)",
    )
    SvgChartGenerator.save_svg(fig4_svg, figures_dir / "fig4_bespoke_operator_comparison.svg")
    print("  -> Fig 4 (Bespoke Operator Comparison) saved.")

    # -------------------------------------------------------------
    # TASK 8: Advanced Domain Preference Fitness Optimization
    # -------------------------------------------------------------
    print("\n[TASK 8] Evaluating Multi-Domain User Preference Optimization...")
    # Preference 1: Cardio & Functional Movement (CV + FM)
    pref_cv_fm = DomainPreferenceFitnessEvaluator(
        preferred_domains={"CV", "FM"},
        preference_penalty_weight=2.5,
        preference_bonus_weight=15.0,
        goal="standard",
    )
    ga_cv_fm = ActiveaGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=pref_cv_fm,
        pop_size=50,
        max_generations=50,
        seed=42,
    )
    best_cv_fm, res_cv_fm, _ = ga_cv_fm.evolve()

    # Preference 2: Strength Development & Muscular Endurance (ST + ME)
    pref_st_me = DomainPreferenceFitnessEvaluator(
        preferred_domains={"ST", "ME"},
        preference_penalty_weight=2.5,
        preference_bonus_weight=15.0,
        goal="standard",
    )
    ga_st_me = ActiveaGeneticAlgorithm(
        activities_pool=kb,
        fitness_evaluator=pref_st_me,
        pop_size=50,
        max_generations=50,
        seed=42,
    )
    best_st_me, res_st_me, _ = ga_st_me.evolve()

    print(f"  -> Preference 1 (CV + FM): {best_cv_fm.to_tuple_list()}")
    print(f"     Preferred Duration Ratio: {res_cv_fm.details.get('preference_ratio', 0.0) * 100:.1f}%")
    print(f"  -> Preference 2 (ST + ME): {best_st_me.to_tuple_list()}")
    print(f"     Preferred Duration Ratio: {res_st_me.details.get('preference_ratio', 0.0) * 100:.1f}%")

    pref_md = [
        "# User Domain Preference Evolved Sessions (Task 8)\n",
        "### Profile 1: Functional Cardiovascular Endurance (Focus: CV & FM)",
        f"- **Total Fitness**: {res_cv_fm.total_fitness:.2f}",
        f"- **Duration**: {best_cv_fm.total_duration:.1f} min",
        f"- **Calories**: {best_cv_fm.total_calories(70.0):.1f} kcal",
        f"- **Adherence to Preferred Domains**: {res_cv_fm.details.get('preference_ratio', 0.0) * 100:.1f}%",
        f"- **Activities**: {best_cv_fm.to_tuple_list()}\n",
        "### Profile 2: Muscular Strength & Conditioning (Focus: ST & ME)",
        f"- **Total Fitness**: {res_st_me.total_fitness:.2f}",
        f"- **Duration**: {best_st_me.total_duration:.1f} min",
        f"- **Calories**: {best_st_me.total_calories(70.0):.1f} kcal",
        f"- **Adherence to Preferred Domains**: {res_st_me.details.get('preference_ratio', 0.0) * 100:.1f}%",
        f"- **Activities**: {best_st_me.to_tuple_list()}",
    ]
    with open(output_dir / "task8_domain_preference_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(pref_md))
    print("  -> Task 8 Preference Results saved.")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTAL TASKS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts and plots saved in: {output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_coursework_pipeline()
