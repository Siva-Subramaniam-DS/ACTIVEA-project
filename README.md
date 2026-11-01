# ACTIVEA

Physical activity planning for DG4COM Computational Intelligence at Aston University.

ACTIVEA searches for one exercise session. A session is a short list of activities, each with a duration in minutes. The search is a generational genetic algorithm. A session is scored with the lecture checks, and its calories come from the coursework activity table scaled to the person's body mass.

## What a session is

The coursework brief writes a session as a set of pairs:

```text
{(jogging, 20), (bodyweight squats, 10)}
```

The program keeps between 2 and 5 different activities. Each duration is between 5 and 45 minutes. If the same activity is proposed twice, the durations are added and then clipped back into that range.

## Calories

The activity table prints kcal per minute for a 70 kg adult. Jogging is 7.35 kcal per minute at 6 MET. For another body mass the brief says to modify that rate:

```text
kcal/min = (kcal/min at 70 kg) × (body mass / 70)
```

The 70 in the formula is the weight the table was printed for. It is not the user's weight. The user's weight is whatever is typed in the body-mass box. At 70 kg the table rate is used as printed. At 80 kg jogging is 7.35 × 80/70 = 8.4 kcal per minute.

Session calories are the sum of those rates times the minutes of each activity. Dream weight and BMI are collected on the form and are not used in this sum.

## Lecture fitness

The score is the lecture rule, an integer from −4 to +4. Each check adds 1 if it passes and subtracts 1 if it fails.

| Check | Pass |
|---|---|
| MET | duration-weighted MET is at least 4 |
| RPE | duration-weighted RPE is between 4 and 7 |
| Heart rate | duration-weighted % of maximum heart rate is between 64 and 80 |

Calories use the goal selected on the page:

| Goal | Calorie band |
|---|---|
| General health | 200–400 kcal |
| Fitness improvement | 300–500 kcal |
| Weight loss | 400–600 kcal |

The lecture code itself uses 300–600 kcal for the calorie check. Domain labels are shown on the page. They do not change the +4 score. A session can score +4 and still leave out strength.

## Genetic algorithm

The loop follows the evolutionary-algorithm lectures:

1. Create a random population of sessions.
2. Score every session with the lecture rule.
3. Pick parents by tournament selection (tournament size 3).
4. Recombine two parents with probability 0.8. Shared activities keep the summed duration, then the set is split into two children.
5. Mutate a child with probability 0.15: shift a duration, replace an activity, insert an activity, or delete an activity.
6. Copy the two best distinct sessions into the next generation.
7. Stop at the generation limit and return the best session found.

## How to run

Python 3.11 is what this project was run with. From the project folder:

```text
pip install numpy pandas scikit-learn joblib matplotlib
```

`numpy`, `pandas`, `scikit-learn` and `joblib` are needed for the saved calorie model, the notebook and the benchmark scripts. The planner page does not need them.

### Planner page

Open `web/index.html` in a browser. Enter body mass, choose a goal, and press **Plan this session**. The page runs the genetic algorithm in the browser and plays the session at one planned minute per second.

`python main.py --web` also serves that page at <http://127.0.0.1:8767/>.

### Menu

```text
python main.py
```

| Choice | What it runs |
|---|---|
| 1 | Search comparison that scores sessions with the saved calorie model, 10 seeds |
| 2 | The same comparison, 3 seeds |
| 3 | Console demo of the genetic algorithm |
| 4 | The planner page on port 8767 |
| 5 | Unit tests |
| 6 | Coursework tasks 1 to 8 |

The same actions from the command line:

```text
python main.py --benchmark
python main.py --quick
python main.py --demo
python main.py --web
python main.py --test
python main.py --experiments
```

Tests on their own:

```text
python -m unittest tests/test_all.py
```

## Calorie model

`ACTIVEA_EDA_and_Model_Optimization.ipynb` trains six regressors on `endomondoHR.csv` to predict **Calories Burn**. The inputs are duration, average heart rate, maximum heart rate, heart-rate standard deviation, speed, elevation gain, distance, sport and gender. Sport and gender are one-hot encoded, which gives 18 features. Each model is scored with 5-fold cross-validation and then on the held-out test set.

| Model | 5-fold CV R² | Train R² | Test R² | Test RMSE (kcal) | Test MAE (kcal) |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.9984 | 0.9990 | 0.9987 | 25.70 | 16.47 |
| Random Forest | 0.9975 | 0.9992 | 0.9982 | 30.09 | 14.58 |
| Decision Tree | 0.9884 | 0.9919 | 0.9890 | 73.86 | 48.58 |
| Linear Regression | 0.9601 | 0.9603 | 0.9597 | 141.19 | 87.56 |
| Ridge (alpha 1.0) | 0.9601 | 0.9603 | 0.9597 | 141.19 | 87.56 |
| Lasso (alpha 0.1) | 0.9601 | 0.9602 | 0.9597 | 141.26 | 87.46 |

The champion is the lowest test RMSE: **Gradient Boosting** (80 trees, learning rate 0.1, max depth 4). It is saved as `Model/best_calorie_model.joblib`. Random Forest has a lower MAE and a higher RMSE, so it is not the champion.

The planner page does not call this model. It uses the activity-table rate above, so the MET on a row and the calories on that row describe the same workout.

## Search methods

`activea/core_ga.py` is the genetic algorithm used for the coursework. The other files are comparisons:

| File | Method |
|---|---|
| `activea/benchmark_sa.py` | Simulated annealing (the nature-inspired comparison) |
| `activea/benchmark_pso.py` | Particle swarm optimisation |
| `activea/benchmark_de.py` | Differential evolution |
| `activea/benchmark_es.py` | (1+1) evolution strategy |
| `activea/benchmark_ils.py` | Iterated local search |
| `activea/benchmark_random.py` | Random search |
| `activea/experiments.py` | Parameter trials and deterministic crowding |
| `activea/advanced.py` | Repair operator, and a penalty for activities outside a chosen domain |

Under the lecture score, many different legal sessions already score +4, so a higher lecture number cannot separate them. The session itself (minutes, calories, domains) is what still differs.

## Project layout

| Path | Role |
|---|---|
| `main.py` | Menu and command-line launcher |
| `web/index.html` | Browser planner |
| `activea/` | Models, fitness, operators, genetic algorithm, comparisons |
| `tests/test_all.py` | Unit tests |
| `exercise_dataset.csv` | Coursework exercise log |
| `endomondoHR.csv` | Wearable table used to train the calorie model |
| `Model/best_calorie_model.joblib` | Saved gradient-boosting pipeline |
| `ACTIVEA_EDA_and_Model_Optimization.ipynb` | Data check and the six regressors |
| `ACTIVEA_Full_Report.docx` | Coursework report, with the training table and charts |
| `ACTIVEA_All_Documents.md` | The markdown write-ups collected in one file |
| `ACTIVEA_File_Guide.xlsx` | One row per project file |
| `results/` | Task tables and charts |
| `Subject PPt/` | Lecture slides |
| `Coursework spec_20267_B1.pdf` | Assessment brief |

## Report and charts

`ACTIVEA_Full_Report.docx` is the written report. Sections 11 and 12 add the notebook training table and these charts from `results/figures/`:

- genetic algorithm against simulated annealing
- diversity with and without deterministic crowding
- the plain genetic algorithm and the repair operator
- mean convergence of the search methods
- convergence when sessions are scored with the saved calorie model

`ACTIVEA_All_Documents.md` joins the task notes, the technical report, the presentation script, the result tables and those same charts.

## Coursework tasks

| Task | Where it is |
|---|---|
| 1. Activity table and data | `activea/data_loader.py`, `exercise_dataset.csv`, `endomondoHR.csv` |
| 2. Representation, operators, fitness | `activea/models.py`, `activea/operators.py`, `activea/fitness.py` |
| 3. Genetic algorithm | `activea/core_ga.py` |
| 4. One nature-inspired comparison | `activea/benchmark_sa.py` |
| 5. Parameter trials | `activea/experiments.py` |
| 6. Diversity | deterministic crowding in `activea/experiments.py` |
| 7. A changed operator | repair operator in `activea/advanced.py` |
| 8. Domain preference | `DomainPreferenceFitnessEvaluator` in `activea/advanced.py` |

## AI use

The sentences in the report were drafted with generative AI in Cursor (Grok). The session form, the jogging row, the three calorie bands, the instruction to scale the 70 kg rate by body mass, and the +1/−1 checks come from the coursework brief and the lecture code. The minutes and calorie totals come from the recorded runs.
