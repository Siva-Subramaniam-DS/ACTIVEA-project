"""ACTIVEA - Main Entry Point Script
Aston University - DG4COM Computational Intelligence Coursework

This file provides a unified launcher for all ACTIVEA modules, benchmarks,
and interactive tools.

Usage:
    python main.py                  # Launches interactive console menu
    python main.py --benchmark      # Runs full 7-algorithm ML benchmark comparison
    python main.py --quick          # Runs quick 7-algorithm ML benchmark (3 seeds)
    python main.py --demo           # Runs interactive live presentation demo
    python main.py --web            # Starts the web planner dashboard on localhost:8767
    python main.py --test           # Runs full unit test suite
    python main.py --experiments    # Runs legacy coursework experiments (Tasks 1-8)
"""

import sys
import os
import subprocess
import argparse

BANNER = r"""
=============================================================================
        ACTIVEA: Physical Activity Planning Evolutionary Algorithm
        Aston University - Computational Intelligence (DG4COM)
=============================================================================
"""

def print_menu():
    print(BANNER)
    print("Select an option to run:")
    print("  [1] Multi-Algorithm ML Benchmark Comparison (All 7 Algorithms, 10 Seeds)")
    print("  [2] Quick Multi-Algorithm Benchmark (All 7 Algorithms, 3 Seeds, ~10s)")
    print("  [3] Interactive Live Presentation Demo (ACSM + ML Calorie Predictions)")
    print("  [4] Web Session Planner Dashboard (Local Browser UI)")
    print("  [5] Run Complete Unit Test Suite (14 Tests)")
    print("  [6] Run Legacy Coursework Experiments Suite (Tasks 1 to 8)")
    print("  [0] Exit")
    print("-" * 77)

def run_script(script_name: str, args: list[str] = None):
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

def main():
    parser = argparse.ArgumentParser(
        description="ACTIVEA - Physical Activity Planning Evolutionary Algorithm Launcher",
        add_help=True
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--benchmark", action="store_true", help="Run full 7-algorithm ML benchmark")
    group.add_argument("-q", "--quick", action="store_true", help="Run quick 7-algorithm ML benchmark")
    group.add_argument("-d", "--demo", action="store_true", help="Run live presentation demo")
    group.add_argument("-w", "--web", action="store_true", help="Launch web planner dashboard server")
    group.add_argument("-t", "--test", action="store_true", help="Run complete unit test suite")
    group.add_argument("-e", "--experiments", action="store_true", help="Run legacy coursework experiment pipeline")

    args = parser.parse_args()

    if args.benchmark:
        run_script("run_ml_comparison.py")
        return
    if args.quick:
        run_script("run_ml_comparison.py", ["--quick"])
        return
    if args.demo:
        run_script("demo.py")
        return
    if args.web:
        run_script("serve_planner.py")
        return
    if args.test:
        subprocess.run([sys.executable, "-m", "unittest", "tests/test_all.py"])
        return
    if args.experiments:
        run_script("run_all_experiments.py")
        return

    # Interactive menu if no arguments provided
    while True:
        print_menu()
        try:
            choice = input("Enter choice [0-6] (Default: 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting ACTIVEA.")
            break

        if choice == "" or choice == "2":
            print("\n>>> Launching Quick Multi-Algorithm Benchmark...\n")
            run_script("run_ml_comparison.py", ["--quick"])
            break
        elif choice == "1":
            print("\n>>> Launching Full Multi-Algorithm Benchmark...\n")
            run_script("run_ml_comparison.py")
            break
        elif choice == "3":
            print("\n>>> Launching Live Presentation Demo...\n")
            run_script("demo.py")
            break
        elif choice == "4":
            print("\n>>> Launching Web Dashboard Planner...\n")
            run_script("serve_planner.py")
            break
        elif choice == "5":
            print("\n>>> Running Unit Tests...\n")
            subprocess.run([sys.executable, "-m", "unittest", "tests/test_all.py"])
            break
        elif choice == "6":
            print("\n>>> Running Legacy Coursework Experiments...\n")
            run_script("run_all_experiments.py")
            break
        elif choice == "0":
            print("Exiting ACTIVEA.")
            break
        else:
            print(f"[!] Invalid selection: '{choice}'. Please enter a number between 0 and 6.\n")

if __name__ == "__main__":
    main()
