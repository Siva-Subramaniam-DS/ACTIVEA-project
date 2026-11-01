import unittest
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution method
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from activea.models import Activity, ExerciseSession, SessionItem, FITNESS_DOMAINS
from activea.data_loader import ExerciseKnowledgeBase, WearableDataLoader
from activea.fitness import ActiveaFitnessEvaluator
from activea.operators import (
    create_random_session,
    initialize_population,
    tournament_selection,
    roulette_selection,
    uniform_crossover,
    single_point_crossover,
    standard_mutation,
)
from activea.core_ga import ActiveaGeneticAlgorithm
from activea.benchmark_sa import SimulatedAnnealingOptimizer
from activea.experiments import CrowdingGeneticAlgorithm
from activea.advanced import BespokeRepairOperator, DomainPreferenceFitnessEvaluator


class TestActiveaModels(unittest.TestCase):
    def setUp(self):
        self.jogging = Activity(
            name="Jogging",
            calories_per_min_70kg=7.35,
            met=6.0,
            target_rpe=5.0,
            target_hr_min=70.0,
            target_hr_max=77.0,
            domains={"CV", "ME", "FM"}
        )
        self.squats = Activity(
            name="Bodyweight squats",
            calories_per_min_70kg=4.90,
            met=4.0,
            target_rpe=6.0,
            target_hr_min=64.0,
            target_hr_max=70.0,
            domains={"ST", "ME", "BA", "FM"}
        )

    def test_activity_weight_scaling(self):
        cal_70 = self.jogging.calories_per_min(70.0)
        self.assertAlmostEqual(cal_70, 7.35, places=2)
        # Scaled from the 70 kg table rate: 7.35 * 80 / 70 = 8.4
        cal_80 = self.jogging.calories_per_min(80.0)
        self.assertAlmostEqual(cal_80, 8.4, places=2)

    def test_exercise_session_metrics(self):
        session = ExerciseSession()
        session.add_activity(self.jogging, 20.0)
        session.add_activity(self.squats, 10.0)

        self.assertEqual(session.total_duration, 30.0)
        # Calories = 20 * 7.35 + 10 * 4.90 = 147.0 + 49.0 = 196.0
        self.assertAlmostEqual(session.total_calories(70.0), 196.0, places=2)
        # Weighted MET = (20 * 6.0 + 10 * 4.0) / 30 = 160 / 30 = 5.333
        self.assertAlmostEqual(session.weighted_met, 5.333, places=2)
        # Domains covered should include CV, ME, FM, ST, BA
        self.assertTrue({"CV", "ST", "ME", "BA", "FM"}.issubset(session.domains_covered))


class TestActiveaFitness(unittest.TestCase):
    def setUp(self):
        self.kb = ExerciseKnowledgeBase.get_default_activities()
        self.evaluator = ActiveaFitnessEvaluator(goal="standard")

    def test_lecture_baseline(self):
        session = ExerciseSession()
        session.add_activity(self.kb["Running"], 35.0)  # ~367 kcal, 8.5 MET
        score = self.evaluator.evaluate_lecture_baseline(session)
        self.assertIsInstance(score, int)

    def test_continuous_evaluator(self):
        session = ExerciseSession()
        session.add_activity(self.kb["Jogging"], 30.0)
        session.add_activity(self.kb["Bodyweight squats"], 20.0)
        res = self.evaluator.evaluate(session)
        self.assertGreater(res.total_fitness, 0.0)
        self.assertGreater(res.calorie_score, 0.0)
        self.assertGreater(res.domain_score, 0.0)

    def test_empty_session_penalty(self):
        empty_session = ExerciseSession()
        res = self.evaluator.evaluate(empty_session)
        self.assertEqual(res.total_fitness, 0.0)


class TestGeneticOperators(unittest.TestCase):
    def setUp(self):
        self.kb = ExerciseKnowledgeBase.get_default_activities()

    def test_population_initialization(self):
        pop = initialize_population(pop_size=10, activities_pool=self.kb)
        self.assertEqual(len(pop), 10)
        for ind in pop:
            self.assertGreaterEqual(len(ind), 2)
            self.assertGreater(ind.total_duration, 0.0)

    def test_crossover_and_mutation(self):
        pop = initialize_population(pop_size=2, activities_pool=self.kb)
        p1, p2 = pop[0], pop[1]
        c1, c2 = uniform_crossover(p1, p2)
        self.assertGreaterEqual(len(c1), 1)
        self.assertGreaterEqual(len(c2), 1)

        mutated = standard_mutation(c1, self.kb, mutation_rate=0.5)
        self.assertIsInstance(mutated, ExerciseSession)


class TestOptimizationEngines(unittest.TestCase):
    def setUp(self):
        self.kb = ExerciseKnowledgeBase.get_default_activities()

    def test_core_ga_execution(self):
        ga = ActiveaGeneticAlgorithm(
            activities_pool=self.kb,
            pop_size=15,
            max_generations=10,
            seed=42
        )
        best_ind, best_res, history = ga.evolve()
        self.assertIsNotNone(best_ind)
        self.assertGreater(best_res.total_fitness, 50.0)
        self.assertEqual(len(history), 11)

    def test_simulated_annealing(self):
        sa = SimulatedAnnealingOptimizer(
            activities_pool=self.kb,
            total_evaluations=150,
            seed=42
        )
        best_ind, best_res, history = sa.optimize()
        self.assertIsNotNone(best_ind)
        self.assertGreater(best_res.total_fitness, 40.0)

    def test_crowding_ga(self):
        cga = CrowdingGeneticAlgorithm(
            activities_pool=self.kb,
            pop_size=16,
            max_generations=10,
            seed=42
        )
        best_ind, best_res, history = cga.evolve()
        self.assertIsNotNone(best_ind)
        self.assertGreater(best_res.total_fitness, 50.0)


class TestAdvancedTasks(unittest.TestCase):
    def setUp(self):
        self.kb = ExerciseKnowledgeBase.get_default_activities()

    def test_task7_bespoke_operator(self):
        operator = BespokeRepairOperator(self.kb)
        # Create a low-calorie deficient session
        low_cal_session = ExerciseSession()
        low_cal_session.add_activity(self.kb["Yoga"], 10.0)  # ~30 kcal
        repaired = operator.repair_and_mutate(low_cal_session)
        # Repaired session calories should be increased
        self.assertGreater(repaired.total_calories(70.0), low_cal_session.total_calories(70.0))

    def test_task8_domain_preference(self):
        # Focus solely on CV and FM
        pref_eval = DomainPreferenceFitnessEvaluator(preferred_domains={"CV", "FM"})
        # CV & FM session
        cv_session = ExerciseSession()
        cv_session.add_activity(self.kb["Jogging"], 30.0)  # Has CV and FM
        cv_session.add_activity(self.kb["Running"], 20.0)  # Has CV and FM
        res_cv = pref_eval.evaluate(cv_session)

        # ST & BA session (lacks CV and FM)
        st_session = ExerciseSession()
        st_session.add_activity(self.kb["Plank"], 30.0)  # ST, ME, BA
        st_session.add_activity(self.kb["Bodyweight squats"], 20.0)  # ST, ME, BA, FM
        res_st = pref_eval.evaluate(st_session)

        # The preferred CV session should achieve higher preference score / lower penalty
        self.assertLess(res_cv.preference_penalty, res_st.preference_penalty)


class TestRemainingMLAlgorithms(unittest.TestCase):
    """Verifies that the ML model evaluator and all 6 remaining metaheuristics execute correctly."""
    def setUp(self):
        from evaluate_with_model import MLSavedModelFitnessEvaluator
        self.evaluator = MLSavedModelFitnessEvaluator(goal="fitness_improvement")

    def test_ml_evaluator(self):
        session = [("Jogging", 15.0), ("Bodyweight squats", 12.0)]
        fit, det = self.evaluator.evaluate(session)
        self.assertGreater(fit, 0.0)
        self.assertGreater(det["predicted_calories"], 0.0)
        self.assertIn("CV", det["domains_covered"])

    def test_remaining_algorithms_smoke(self):
        from remaining_algorithms import (
            RandomSearchML,
            SimulatedAnnealingML,
            IteratedLocalSearchML,
            ParticleSwarmML,
            DifferentialEvolutionML,
            BespokeRepairGA_ML,
            StandardGeneticAlgorithmML
        )
        algos = [
            RandomSearchML(self.evaluator, seed=1),
            SimulatedAnnealingML(self.evaluator, seed=1),
            IteratedLocalSearchML(self.evaluator, seed=1),
            ParticleSwarmML(self.evaluator, seed=1),
            DifferentialEvolutionML(self.evaluator, seed=1),
            BespokeRepairGA_ML(self.evaluator, seed=1),
            StandardGeneticAlgorithmML(self.evaluator, seed=1),
        ]
        for a in algos:
            res = a.optimize(budget=20)
            self.assertGreater(res["best_fitness"], 0.0)
            self.assertGreaterEqual(len(res["best_solution"]), 2)
            self.assertLessEqual(len(res["best_solution"]), 5)


if __name__ == "__main__":
    unittest.main()
