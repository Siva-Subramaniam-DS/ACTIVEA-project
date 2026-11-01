"""ACTIVEA - Head-to-Head Metaheuristic Optimization Benchmark Runner
Evaluates all 7 optimization algorithms coupled with the Champion ML Caloric Regressor ('Model/best_calorie_model.joblib').

Adheres to rigorous scientific benchmarking methodology (Lecture slides 'Algorithm Comparison'):
1. Standardized Evaluation Budget (Budget = 2,500 evaluations per run)
2. Multi-Seed Replication (e.g. 10 to 30 independent stochastic trials)
3. Formal Hypothesis Testing (Welch's two-tailed unequal-variance t-test vs. Standard GA)
4. Multi-Line Convergence Trajectory Analysis
5. Publication-grade SVG visualization and Markdown table generation

Usage:
    python run_ml_comparison.py                 # Default: 10 seeds, 2500 budget
    python run_ml_comparison.py --quick         # Fast test: 3 seeds, 500 budget
    python run_ml_comparison.py --seeds 30      # Full rigor: 30 seeds, 2500 budget
"""

from __future__ import annotations
import sys
import os
import math
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from evaluate_with_model import MLSavedModelFitnessEvaluator
from remaining_algorithms import (
    RandomSearchML,
    SimulatedAnnealingML,
    IteratedLocalSearchML,
    ParticleSwarmML,
    DifferentialEvolutionML,
    BespokeRepairGA_ML,
    StandardGeneticAlgorithmML
)
from activea.visualizer import SvgChartGenerator


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formats a pandas DataFrame into a GitHub-flavored Markdown table without external libraries."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(cols))]) + " |"
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


# =============================================================================
# STATISTICAL HYPOTHESIS TESTING (PURE PYTHON WELCH'S T-TEST)
# =============================================================================

def compute_mean_and_std(values: List[float]) -> Tuple[float, float]:
    """Computes sample mean and sample standard deviation (unbiased, ddof=1)."""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0, 0.0)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3.0e-7) -> float:
    """Continued fraction approximation for incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < eps:
            break
    return h


def ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    else:
        return 1.0 - front * betacf(b, a, 1.0 - x) / b


def welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float, float]:
    """Computes Welch's two-sample unequal-variance t-test.
    Returns: (t_statistic, degrees_of_freedom, two_tailed_p_value)
    """
    n1, n2 = len(sample1), len(sample2)
    m1, s1 = compute_mean_and_std(sample1)
    m2, s2 = compute_mean_and_std(sample2)

    v1 = (s1 ** 2) / n1
    v2 = (s2 ** 2) / n2
    denom = math.sqrt(v1 + v2)

    if denom < 1e-12:
        return 0.0, float(n1 + n2 - 2), 1.0

    t_stat = (m1 - m2) / denom
    df = ((v1 + v2) ** 2) / (((v1 ** 2) / (n1 - 1)) + ((v2 ** 2) / (n2 - 1)))
    x = df / (df + (t_stat ** 2))
    p_val = ibeta(df / 2.0, 0.5, x)
    return t_stat, df, p_val


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

ALGORITHM_FACTORIES = [
    ("Random Search", RandomSearchML),
    ("Simulated Annealing", SimulatedAnnealingML),
    ("Iterated Local Search", IteratedLocalSearchML),
    ("Particle Swarm Optimization", ParticleSwarmML),
    ("Differential Evolution", DifferentialEvolutionML),
    ("Bespoke Repair GA", BespokeRepairGA_ML),
    ("Standard Genetic Algorithm", StandardGeneticAlgorithmML),
]


def run_benchmark(
    num_seeds: int = 10,
    budget: int = 2500,
    goal: str = "fitness_improvement",
    checkpoint_interval: int = 50
) -> Dict[str, Any]:
    print("=" * 80)
    print("ACTIVEA - HEAD-TO-HEAD ALGORITHM COMPARISON (POWERED BY ML MODEL)")
    print(f"Goal Setting       : {goal.replace('_', ' ').title()}")
    print(f"Independent Seeds  : {num_seeds} trials per algorithm")
    print(f"Evaluation Budget  : {budget:,} evaluations per run")
    print("=" * 80)

    evaluator = MLSavedModelFitnessEvaluator(goal=goal)
    seeds = [42 + i * 17 for i in range(num_seeds)]

    raw_results: Dict[str, Dict[str, Any]] = {}
    checkpoints = sorted(list(range(checkpoint_interval, budget + 1, checkpoint_interval)))
    if budget not in checkpoints:
        checkpoints.append(budget)

    total_start_time = time.time()

    for name, factory in ALGORITHM_FACTORIES:
        print(f"\n[Running Benchmark] -> {name} ({num_seeds} seeds)...", flush=True)
        fits = []
        runtimes = []
        trajectories = []
        best_overall_fit = -1.0
        best_overall_sol = None
        best_overall_det = None

        t_algo_start = time.time()
        for idx, seed in enumerate(seeds, 1):
            algo_instance = factory(evaluator=evaluator, seed=seed)
            res = algo_instance.optimize(budget=budget, checkpoint_interval=checkpoint_interval)

            fit = res["best_fitness"]
            fits.append(fit)
            runtimes.append(res["runtime_sec"])

            # Map history to standard checkpoints
            hist_df = res["history"]
            eval_to_fit = dict(zip(hist_df["eval"], hist_df["best_fitness"]))
            # Forward-fill checkpoints
            curr_val = hist_df["best_fitness"].iloc[0]
            curve = []
            for cp in checkpoints:
                if cp in eval_to_fit:
                    curr_val = eval_to_fit[cp]
                curve.append(curr_val)
            trajectories.append(curve)

            if fit > best_overall_fit:
                best_overall_fit = fit
                best_overall_sol = res["best_solution"]
                best_overall_det = res["best_details"]

        algo_elapsed = time.time() - t_algo_start
        mean_fit, std_fit = compute_mean_and_std(fits)
        mean_time, std_time = compute_mean_and_std(runtimes)

        # Average convergence curve across seeds
        avg_trajectory = [
            round(sum(trajectories[s][c] for s in range(num_seeds)) / num_seeds, 2)
            for c in range(len(checkpoints))
        ]

        raw_results[name] = {
            "mean_fitness": round(mean_fit, 2),
            "std_fitness": round(std_fit, 2),
            "min_fitness": round(min(fits), 2),
            "max_fitness": round(max(fits), 2),
            "all_fits": [round(f, 2) for f in fits],
            "mean_runtime_sec": round(mean_time, 2),
            "total_time_sec": round(algo_elapsed, 2),
            "best_solution": best_overall_sol,
            "best_details": best_overall_det,
            "convergence_curve": avg_trajectory
        }

        print(f"  * Mean Fitness : {mean_fit:.2f} +/- {std_fit:.2f} (Min: {min(fits):.2f}, Max: {max(fits):.2f})")
        print(f"  * Mean Runtime : {mean_time:.2f}s | Finished in {algo_elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # STATISTICAL HYPOTHESIS TESTING (VS. STANDARD GA)
    # -------------------------------------------------------------------------
    ga_fits = raw_results["Standard Genetic Algorithm"]["all_fits"]
    comparison_rows = []

    for name in [n for n, _ in ALGORITHM_FACTORIES]:
        data = raw_results[name]
        m_fit, s_fit = data["mean_fitness"], data["std_fitness"]
        min_f, max_f = data["min_fitness"], data["max_fitness"]
        t_sec = data["mean_runtime_sec"]

        if name == "Standard Genetic Algorithm":
            p_val_str = "Baseline (N/A)"
            sig_str = "Reference Baseline"
        else:
            t_stat, df, p_val = welch_t_test(data["all_fits"], ga_fits)
            p_val_str = f"{p_val:.4f}"
            if p_val < 0.05:
                if m_fit > raw_results["Standard Genetic Algorithm"]["mean_fitness"]:
                    sig_str = "Superior (p < 0.05) *"
                else:
                    sig_str = "Inferior (p < 0.05) *"
            else:
                sig_str = "Equivalent (p >= 0.05)"

        comparison_rows.append({
            "raw_mean": m_fit,
            "Algorithm": name,
            "Mean Fitness +/- Std": f"{m_fit:.2f} +/- {s_fit:.2f}",
            "Min Fit": f"{min_f:.2f}",
            "Max Fit": f"{max_f:.2f}",
            "Runtime (s)": f"{t_sec:.2f}s",
            "p-value (vs GA)": p_val_str,
            "Significance": sig_str
        })

    # Output formatting sorted by numeric mean fitness
    df_results = (
        pd.DataFrame(comparison_rows)
        .sort_values(by="raw_mean", ascending=False)
        .drop(columns=["raw_mean"])
        .reset_index(drop=True)
    )

    print("\n" + "=" * 90)
    print("FINAL 30-SEED STATISTICAL BENCHMARK COMPARISON TABLE (vs. STANDARD GA)")
    print("=" * 90)
    print(df_results.to_string(index=False))
    print("=" * 90)

    # -------------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Save Markdown Table
    md_table_path = output_dir / "ml_benchmark_comparison_table.md"
    md_content = [
        "# Head-to-Head Algorithm Comparison (Powered by Champion ML Caloric Model)\n\n",
        f"**Goal Profile:** {goal.replace('_', ' ').title()}  \n",
        f"**Evaluation Budget:** {budget:,} evaluations per run  \n",
        f"**Replications:** {num_seeds} independent stochastic trials  \n",
        f"**Hypothesis Testing:** Welch's two-sample unequal-variance t-test against Standard GA  \n\n",
        dataframe_to_markdown(df_results),
        "\n\n*Note: An asterisk (*) indicates statistically significant difference at alpha = 0.05 level.*"
    ]
    md_table_path.write_text("".join(md_content), encoding="utf-8")
    print(f"\n[Artifact Saved] Markdown table -> {md_table_path}")

    # 2. Save Convergence SVG Plot
    series_data = {name: raw_results[name]["convergence_curve"] for name, _ in ALGORITHM_FACTORIES}
    svg_plot = SvgChartGenerator.generate_convergence_plot(
        series_data=series_data,
        title=f"Head-to-Head Algorithm Convergence ({budget:,} Evaluations, {num_seeds} Seeds)",
        x_label="Evaluation Budget Checkpoint",
        y_label="Fitness Score [0 - 100]"
    )
    svg_path = figures_dir / "ml_benchmark_convergence.svg"
    svg_path.write_text(svg_plot, encoding="utf-8")
    print(f"[Artifact Saved] Convergence SVG plot -> {svg_path}")

    # 3. Save Raw JSON Results
    json_path = output_dir / "ml_benchmark_raw_results.json"
    json_path.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")
    print(f"[Artifact Saved] Raw benchmark data -> {json_path}")

    total_time = time.time() - total_start_time
    print(f"\nFull benchmarking pipeline completed successfully in {total_time:.1f} seconds ({total_time/60:.2f} min).")
    return raw_results


def main():
    parser = argparse.ArgumentParser(description="ACTIVEA ML-Coupled Algorithm Benchmark Runner")
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent random seeds (default: 10)")
    parser.add_argument("--budget", type=int, default=2500, help="Evaluation budget per algorithm (default: 2500)")
    parser.add_argument("--goal", type=str, default="fitness_improvement", choices=["fitness_improvement", "general_health", "weight_loss"])
    parser.add_argument("--quick", action="store_true", help="Run a quick verification test (3 seeds, 500 budget)")
    args = parser.parse_args()

    if args.quick:
        run_benchmark(num_seeds=3, budget=500, goal=args.goal, checkpoint_interval=50)
    else:
        run_benchmark(num_seeds=args.seeds, budget=args.budget, goal=args.goal, checkpoint_interval=50)


if __name__ == "__main__":
    main()
