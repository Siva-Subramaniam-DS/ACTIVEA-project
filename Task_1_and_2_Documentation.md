# DG4COM Computational Intelligence — Coursework Brief: Tasks 1 & 2
**Module:** DG4COM Computational Intelligence | Aston University  
**Project:** ACTIVEA (Physical Activity Planning Evolutionary Algorithm)  
**Deliverable Scope:** Preparation & Technical Documentation for **Core Task 1 (Data Setup)** and **Core Task 2 (Algorithm Setup & Model Formulation)**  

---

## 1. Executive Summary

This document fulfills the requirements for **Core Task 1** and **Core Task 2** as outlined in the official Aston University coursework specification and class teaching sessions.

* **Task 1 (Data Setup):** Establishes the exercise quality knowledge base comprising standardized physical activities with complete exercise-science intensity metrics (calibrated for a 70 kg individual using ACSM guidelines), alongside the partitioning and stratification of the empirical wearable telemetry dataset (`wearables_monitoring_data.csv`).
* **Task 2 (Algorithm Setup & Model Build):** Formulates the evolutionary algorithm architecture: chromosomal representation, selection/recombination/mutation operators, termination criteria, and both the discrete lecture baseline fitness formulation (`fit.py.txt`) and the advanced continuous multi-attribute fitness function.

---

## 2. Task 1: Data Setup & Knowledge Base

### 2.1 Completed Exercise Quality Knowledge Base

As instructed in the coursework brief (Pages 2–3), candidate physical activities are characterized across metabolic expenditure, perceived exertion, cardiovascular intensity, and supported fitness domains.

| Activity | Calories/min (70 kg) | MET | Target RPE (1–10) | Target HR [% MHR] | Fitness Domain(s) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Jogging** | **7.35** | **6.0** | **5.0** | **70–77%** | **CV, ME, FM** |
| **Bodyweight squats** | **4.90** | **4.0** | **6.0** | **64–70%** | **ST, ME, BA, FM** |
| Cycling (moderate) | 6.80 | 5.5 | 4.0 | 65–75% | CV, ME |
| Plank | 3.50 | 3.0 | 7.0 | 60–68% | ST, ME, BA |
| Yoga | 3.00 | 2.5 | 3.0 | 55–65% | FL, BA, FM |
| Rope skipping | 11.00 | 9.0 | 8.0 | 78–88% | CV, AG, ME, FM |
| Running | 10.50 | 8.5 | 7.0 | 77–85% | CV, ME, FM |
| Brisk walking | 4.10 | 3.5 | 3.0 | 55–65% | CV, FM |
| Pushups & Calisthenics | 5.60 | 4.5 | 6.0 | 65–75% | ST, ME, FM |
| Agility drills & Burpees | 9.80 | 8.0 | 8.0 | 80–90% | CV, AG, ST, FM |

---

### 2.2 Exercise-Science Metric Definitions & Citations

1. **Calories Burned per Minute ($\text{kcal}\cdot\text{min}^{-1}$):**
   Caloric consumption for a standard 70 kg adult is computed directly from METs using the recognized American College of Sports Medicine (ACSM, 2018) metabolic equation:
   $$\text{Calories/min} = \frac{\text{MET} \times 3.5 \times \text{Weight}_{\text{kg}}}{200}$$
   For a 70 kg individual: $\text{Calories/min} = \text{MET} \times 1.225$.
   *Example:* Jogging ($\text{MET} = 6.0$) burns $6.0 \times 1.225 = 7.35\text{ kcal/min}$.

2. **Metabolic Equivalent of Task (MET):**
   Expresses the energy cost of physical activity as a multiple of resting metabolic rate ($1\text{ MET} \approx 3.5\text{ ml } \text{O}_2\cdot\text{kg}^{-1}\cdot\text{min}^{-1}$). Values are cross-referenced with the *2011 Compendium of Physical Activities* (Ainsworth et al., 2011; Jetté et al., 1990).

3. **Target Rate of Perceived Exertion (RPE):**
   Quantified using the Borg CR10 Scale (Borg, 1998), spanning from 1 (very light) to 10 (maximal exertion). Target RPE sets the planned effort ceiling for safely sustaining the workout.

4. **Target Heart Rate Range (% MHR):**
   Percentage of Maximum Heart Rate ($MHR \approx 220 - \text{Age}$ or Karvonen heart-rate reserve formula). Corresponds to ACSM aerobic training zones:
   * Moderate aerobic endurance: $64\%\text{--}76\%$ MHR.
   * Vigorous cardiorespiratory conditioning: $77\%\text{--}93\%$ MHR.

5. **Seven Fitness Domains:**
   * **CV (Cardiovascular Fitness):** Sustained aerobic oxygen delivery (e.g., Jogging, Cycling, Running).
   * **ST (Strength Development):** High-force muscular resistance (e.g., Squats, Pushups).
   * **ME (Muscular Endurance):** Sustained submaximal contractions (e.g., Plank, Squats).
   * **FL (Flexibility/Mobility):** Full usable range of joint motion (e.g., Yoga).
   * **BA (Balance & Stability):** Static and dynamic core stability (e.g., Plank, Yoga).
   * **AG (Agility):** Rapid multidirectional deceleration/acceleration (e.g., Rope skipping, Agility drills).
   * **FM (Functional Movement):** Natural compound movement mechanics (e.g., Squats, Walking, Burpees).

---

### 2.3 Wearable Telemetry Dataset Partitioning

The empirical dataset (`wearables_monitoring_data.csv`) contains $10,000$ physiological observation records from wearable sensors across 6 activity regimes:
* `Lying`
* `Sitting`
* `Self Pace walk`
* `Running 3 METs`
* `Running 5 METs`
* `Running 7 METs`

To guarantee rigorous model validation without data contamination:
* **Stratified Split:** Data is stratified proportionally across the 6 physical activity classes.
* **Training Partition (80%):** $N_{\text{train}} = 7,998$ records, used to calibrate baseline user profiles and Karvonen relative intensities.
* **Testing Partition (20%):** $N_{\text{test}} = 2,002$ records, reserved for out-of-sample evaluation of evolved workout recommendations.

---

## 3. Task 2: Algorithm Setup & Model Formulation

### 3.1 Algorithm Selection: Genetic Algorithm (GA)

A **Genetic Algorithm (GA)** was selected as the foundational evolutionary optimization paradigm because exercise session planning is a combinatorial subset-and-parameter allocation problem (selecting $m$ activities and assigning continuous execution durations). GAs naturally balance global exploration across activity combinations with local exploitation of duration allocations.

---

### 3.2 Chromosomal Representation

A workout session chromosome is structured as a variable-length sequence of tuples:
$$\mathcal{S} = \left[ (a_1, d_1), (a_2, d_2), \dots, (a_m, d_m) \right]$$
* **Activity Allele ($a_i \in \mathcal{A}$):** References a unique activity from the knowledge base.
* **Duration Allele ($d_i \in [5.0, 45.0]$ minutes):** Execution duration in minutes.
* **Chromosome Length ($m \in [2, 5]$):** Constrained to prevent single-activity monotony ($m \ge 2$) while avoiding over-fragmentation ($m \le 5$).

---

### 3.3 Genetic Operators

1. **Selection:**
   * **Tournament Selection ($k = 3$):** Randomly draws 3 individuals from the population and selects the highest-fitness candidate. Controls selection pressure while avoiding premature convergence.
   * **Elitism ($E = 2$):** Unconditionally preserves the top 2 best solutions from generation $t$ into generation $t+1$, ensuring strict monotonic fitness progression.

2. **Recombination (Crossover):**
   * **Uniform Recombination with Deduplication:** For two selected parent sessions $P_1$ and $P_2$, activities are merged and partitioned. Duplicate occurrences of the same activity are aggregated by summing durations:
     $$(a_i, d_{i,1}) + (a_i, d_{i,2}) \implies (a_i, d_{i,1} + d_{i,2})$$

3. **Mutation ($p_m = 0.15$):**
   * **Duration Perturbation:** Adjusts duration alleles by small increments $\Delta d \in \{-10, -5, -2, +2, +5, +10\}\text{ min}$, clipped to $[5.0, 45.0]$.
   * **Activity Substitution:** Replaces an activity with an unselected activity from the library $\mathcal{A}$.
   * **Structural Addition/Deletion:** Dynamically expands or contracts the session length within $[2, 5]$ items.

---

### 3.4 Fitness Function Formulation

#### Model A: The Lecture Baseline (`fit.py.txt` / Code 2)
In the lecture slides, the baseline fitness is evaluated using a discrete step scoring model ($+1$ for satisfying targets, $-1$ otherwise):

```python
calc_fitness(actv_session) returns fit_val := 0:
    # 1. Caloric constraint (healthy range 300 to 600 calories)
    if (calories >= 300 AND calories <= 600):
        fit_val = fit_val + 1
    else:
        fit_val = fit_val - 1

    # 2. MET intensity constraint
    if (MET >= 4.0):
        fit_val = fit_val + 1
    else:
        fit_val = fit_val - 1

    # 3. Target RPE constraint
    if (RPE >= 4.0 AND RPE <= 7.0):
        fit_val = fit_val + 1
    else:
        fit_val = fit_val - 1

    # 4. Target Heart Rate % constraint
    if (HR >= 64.0 AND HR <= 80.0):
        fit_val = fit_val + 1
    else:
        fit_val = fit_val - 1
```

*Theoretical Analysis:* While straightforward, this stepwise scoring produces a discrete plateau landscape (fitness values restricted to integers $\{-4, -2, 0, +2, +4\}$) with zero gradient information for chromosomes near the boundary.

#### Model B: Advanced Continuous Multi-Attribute Fitness
To guide evolutionary convergence smoothly, ACTIVEA scales the objective into a continuous 100-point normalized function:

$$F(\mathcal{S}) = \max\left(0, \; w_{\text{cal}} f_{\text{cal}} + w_{\text{met}} f_{\text{met}} + w_{\text{rpe}} f_{\text{rpe}} + w_{\text{hr}} f_{\text{hr}} + w_{\text{dom}} f_{\text{dom}} - P_{\text{dur}}(\mathcal{S}) - P_{\text{struct}}(\mathcal{S})\right)$$

Where the normalized component scores ($f_i \in [0, 1]$) are:
* **Caloric Score ($w_{\text{cal}} = 30$):**
  $$f_{\text{cal}} = \max\left(0, \; 1 - \frac{|C_{\text{total}} - C_{\text{target}}|}{C_{\max} - C_{\min}}\right)$$
* **MET Intensity Score ($w_{\text{met}} = 15$):**
  $$f_{\text{met}} = \max\left(0, \; 1 - \frac{|\overline{\text{MET}} - \text{MET}_{\text{target}}|}{\Delta \text{MET}}\right)$$
* **RPE Effort Score ($w_{\text{rpe}} = 15$):**
  $$f_{\text{rpe}} = \max\left(0, \; 1 - \frac{|\overline{\text{RPE}} - \text{RPE}_{\text{target}}|}{\Delta \text{RPE}}\right)$$
* **Heart Rate Alignment ($w_{\text{hr}} = 15$):**
  $$f_{\text{hr}} = \max\left(0, \; 1 - \frac{|\overline{\text{HR}} - \text{HR}_{\text{target}}|}{\Delta \text{HR}}\right)$$
* **Domain Diversity ($w_{\text{dom}} = 25$):**
  $$f_{\text{dom}} = \frac{|\text{Domains Covered}|}{7}$$
* **Penalty Function $P(\mathcal{S})$:** Penalizes sessions outside total duration bounds $[20, 75]$ minutes ($1.5\text{ pts/min}$) and penalizes mono-activity routines ($10\text{ pts}$).

---

### 3.5 Termination Criteria

In accordance with `Code 1 (DG4COMwk1psd_ACTIVEA.py.txt)`:
$$\text{Termination} := \left(\text{Fitness}(\mathcal{S}_{\text{best}}) \ge \tau\right) \lor \left(\text{Generation} \ge G_{\max}\right)$$
* **Quality Threshold ($\tau = 95.0$):** Prematurely halts evolution if a near-perfect workout is synthesized.
* **Maximum Generation Budget ($G_{\max} = 60$):** Guarantees computational efficiency (execution under $0.5$ seconds per run).

---

## 4. Codebase Architecture & File Mapping

The specifications for Tasks 1 and 2 are fully realized in the codebase:

| Coursework Component | Source Code Implementation |
|---|---|
| **Activity & Session Models** | [`activea/models.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/activea/models.py) |
| **Task 1: Activity Table & Dataset Loader** | [`activea/data_loader.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/activea/data_loader.py) |
| **Task 2: Fitness Evaluators (Baseline & Continuous)** | [`activea/fitness.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/activea/fitness.py) |
| **Task 2: Selection, Crossover & Mutation** | [`activea/operators.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/activea/operators.py) |
| **Task 2 / 3: Evolutionary Execution Loop** | [`activea/core_ga.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/activea/core_ga.py) |
| **Verification & Unit Tests** | [`tests/test_all.py`](file:///c:/Aston%20Class%20Work/Computational%20Intelligence/Task%201/tests/test_all.py) |
