# ACTIVEA: Presentation Slides & Live Demonstration Script
**Module:** DG4COM Computational Intelligence — Aston University  
**Assessment:** Oral Presentation & Live Software Demonstration (50% Weighting)  
**Allocated Duration:** 8 minutes (±10%, 7:12 to 8:48) + 5 minutes Q&A  
**Target Audience:** General Technical Audience (Accessible, visual, plain English)

---

## Slide Deck Overview & Timing Schedule

| Slide # | Title | Key Topic | Target Time | Cumulative Time |
|---|---|---|---|---|
| **Slide 1** | Title & Introduction | Problem Overview & Practical Challenge | 0:45 | 0:45 |
| **Slide 2** | The Core Solution: Evolutionary Algorithm | Chromosome Design & Genetic Analogy | 1:00 | 1:45 |
| **Slide 3** | Formulating Exercise Quality (Fitness) | Multi-Attribute Objective Formulation | 1:00 | 2:45 |
| **Slide 4** | **Live Demonstration** | Running `demo.py` & Analyzing Outputs | 1:30 | 4:15 |
| **Slide 5** | Comparative Benchmark: GA vs. Simulated Annealing | Recombination vs. Single-Point Search | 1:00 | 5:15 |
| **Slide 6** | Systematic Parameter Tuning | Sensitivity, Population Size & Mutation | 0:45 | 6:00 |
| **Slide 7** | Preserving Diversity: Deterministic Crowding | Avoiding Monotony & Premature Trap | 0:45 | 6:45 |
| **Slide 8** | Advanced Extensions: Tailored Preferences & Heuristic Repair | Distinction Tasks 7 & 8 Results | 0:45 | 7:30 |
| **Slide 9** | Summary & Impact | Practical Takeaways & Sports Science Link | 0:30 | **8:00** |
| **Q&A** | Defense Strategy | Handling the 5-Minute Tutor Panel | — | +5:00 |

---

## Detailed Slide Outlines & Verbatim Presenter Commentary

---

### Slide 1: Title & The Challenge of Exercise Planning
**Visual Elements:**
- Title: *ACTIVEA — Evolving Personalized Physical Exercise Sessions using Computational Intelligence*
- Split graphic: Left = Confused individual trying to balance calories, heart rate, and fatigue; Right = Clean, automated schedule balancing 7 fitness dimensions.
- Key Metrics Banner: *Caloric Range (300–600 kcal) | 7 Fitness Domains | Physiological Safety*.

**Spoken Commentary (0:00 – 0:45):**
> "Good morning everyone. Designing an effective workout session is deceptively difficult. If you ask a fitness tracker for a plan, it might suggest running for 45 minutes. But that only trains your cardiovascular stamina, leaving out muscular strength, mobility, and balance.
> 
> Sports science guidelines—such as those from the ACSM—recommend that a healthy workout must balance several competing criteria: burning between 300 and 600 calories, keeping heart rate in a safe, productive zone, and stimulating diverse fitness domains like endurance, agility, and flexibility. 
> 
> Today, I present **ACTIVEA**—an Evolutionary Algorithm designed to automatically search through millions of possible workout combinations and synthesize the optimal physical exercise session in less than a tenth of a second."

---

### Slide 2: The Core Concept — How the Algorithm Works
**Visual Elements:**
- Flow diagram of the evolutionary cycle:
  `Random Workout Population -> Fitness Evaluation -> Tournament Selection -> Uniform Crossover -> Multi-Mode Mutation -> Next Generation`.
- Chromosome schematic:
  `[ (Running, 26 min), (Yoga, 23 min), (Burpees, 11 min) ]`
  Showing how genes represent physical activities and continuous duration alleles.

**Spoken Commentary (0:45 – 1:45):**
> "To solve this, we model a workout session as an artificial chromosome—a sequence of activities, each paired with a duration in minutes. 
> 
> Rather than relying on simple trial-and-error, ACTIVEA mimics biological evolution. We start with a diverse population of 50 random workout sessions. 
> In each generation, sessions undergo 'Tournament Selection' where the fittest routines are chosen to reproduce.
> 
> We recombine them using uniform crossover—swapping exercise blocks between two parent routines—and apply mutation to tweak durations or test alternative activities from our exercise library. Over 40 to 50 generations, the population naturally weeds out ineffective routines and converges toward a well-balanced session."

---

### Slide 3: What Makes a Workout 'Healthy'? The Fitness Function
**Visual Elements:**
- Equation breakdown into four intuitive visual dials:
  1. **Caloric Target (30% weight):** Ideal bell-curve centered at 450 kcal.
  2. **Intensity Triad (45% weight):** MET, Perceived Exertion (RPE), and Target Heart Rate %MHR.
  3. **Holistic Domain Coverage (25% weight):** Stimulating CV, ST, ME, FL, BA, AG, and FM.
  4. **Duration Guardrails:** Penalizing sessions shorter than 20 min or longer than 75 min.

**Spoken Commentary (1:45 – 2:45):**
> "In our initial lecture materials, fitness was judged by simple pass/fail checks: add one point if calories are between 300 and 600, subtract one otherwise. While simple, that creates a flat landscape where the algorithm cannot sense whether it is getting closer to the goal.
> 
> We transformed this into a continuous, multi-attribute fitness function scored out of 100 points. 
> First, calories are rewarded on a smooth gradient peaking at 450 calories—the clinical sweet spot.
> Second, we calculate duration-weighted averages for metabolic intensity (MET), perceived exertion, and heart rate, ensuring the user exercises at a safe, moderate-to-vigorous intensity.
> Third, we reward multi-domain support: the more body systems challenged—from strength to flexibility—the higher the score.
> Finally, penalties prevent unrealistic routines, such as extreme two-hour marathons."

---

### Slide 4: Live Software Demonstration
**Visual Elements:**
- Switch to terminal window.
- Execute command:
  ```powershell
  python demo.py
  ```
- Show interactive menu, select Profile 1, display live convergence in ~0.08 seconds, and review the resulting table.

**Spoken Commentary (2:45 – 4:15):**
> "Let us see ACTIVEA in action with a live demonstration. 
> *(Run `python demo.py 1` in the terminal)*
> 
> As you can see on screen, the system loads our verified exercise library and begins the evolutionary search across 50 generations.
> In just **0.07 seconds**, the algorithm converged with a fitness score of **99.1 out of 100**.
> 
> Look at the generated workout:
> 1. 23 minutes of Rope Skipping for high-intensity cardiovascular and agility conditioning;
> 2. 12 minutes of Burpees for dynamic strength and endurance;
> 3. 28 minutes of Yoga for mobility, balance, and post-workout active recovery.
> 
> Total duration is exactly 63 minutes, burning 454.6 calories—virtually spot-on to our 450-calorie target—and successfully hitting **all 7 fitness domains**. This demonstrates that our algorithm creates realistic, safe, and physiologically coherent routines without manual intervention."

---

### Slide 5: Benchmark Comparison — Genetic Algorithm vs. Simulated Annealing
**Visual Elements:**
- Line chart (Fig 1 from results): Convergence Curves showing GA vs. Simulated Annealing.
- Comparison callout box:
  - GA: Reaches 98.88 fitness, **100% 7-domain coverage**.
  - SA: Reaches 99.70 point fitness, but **only 6/7 domains**.

**Spoken Commentary (4:15 – 5:15):**
> "To validate whether an evolutionary algorithm is truly the best approach, we benchmarked ACTIVEA against Simulated Annealing—a classical physics-inspired optimization technique—giving both algorithms the exact same evaluation budget.
> 
> Simulated Annealing ran very quickly and fine-tuned numerical durations effectively. However, it suffered from a key limitation: it frequently got stuck in local optima covering only 5 or 6 fitness domains. Because it modifies only one solution at a time, it struggled to make the coordinated leaps needed to swap multiple exercises.
> 
> In contrast, ACTIVEA's population crossover allowed it to splice separate building blocks together—pairing a cardio block with a mobility block—consistently achieving full 7-domain holistic coverage."

---

### Slide 6: Systematic Parameter Tuning (Task 5)
**Visual Elements:**
- Bar chart / Summary table showing parameter sweep across Population Size, Mutation Rate, Crossover Rate, and Tournament Size.
- Highlighting the optimal configuration: `Pop = 50, Mut = 0.15, Cross = 0.60–0.80, Tourn = 3`.

**Spoken Commentary (5:15 – 6:00):**
> "Next, we investigated parameter sensitivity across multiple seeds.
> 
> With a small population of 20, the algorithm suffered from high variance and premature convergence because genetic diversity was lost too quickly.
> A population of 50 proved to be the sweet spot: high solution quality (98.77 ± 0.31) with an execution time under 0.07 seconds.
> We also found that moderate crossover rates around 0.6 to 0.8 worked best; pushing crossover to 0.95 disrupted high-performing exercise combinations."

---

### Slide 7: Preserving Population Diversity (Task 6)
**Visual Elements:**
- Diversity Trajectory Graph (Fig 3 from results): Standard GA vs. Deterministic Crowding.
- Diagram showing parent-child distance tournament mechanism.

**Spoken Commentary (6:00 – 6:45):**
> "A common flaw in genetic algorithms is that the entire population eventually becomes identical copies of a single workout. That is terrible for fitness, because a person does not want to repeat the exact same workout every day.
> 
> To solve this, we implemented **Deterministic Crowding**. Instead of offspring competing against the entire population, each child competes only against the parent it most closely resembles.
> As shown on the graph, while the standard GA loses diversity rapidly, Deterministic Crowding preserves distinct sub-niches. In a single run, it preserves both a cardio-focused routine and a strength-focused routine, providing users with varied, high-quality choices."

---

### Slide 8: Advanced Extensions — Heuristic Repair & User Preferences (Tasks 7 & 8)
**Visual Elements:**
- Graphic showing Bespoke Operator: Caloric gap detection and automatic domain splicing.
- Two Evolved Profiles:
  - *Profile 1 (CV & Functional Focus):* 100% adherence to chosen domains.
  - *Profile 2 (Strength & Endurance Focus):* 100% adherence to chosen domains.

**Spoken Commentary (6:45 – 7:30):**
> "Finally, we implemented two advanced extensions for Distinction-level marks.
> 
> In Task 7, we replaced blind mutation with a **Bespoke Domain-Guided Repair Operator**. If a routine is under-calorie, the operator calculates the exact caloric deficit and scales duration or inserts a high-MET exercise. This accelerated convergence speed by 40%, reaching near-optimal fitness in just 12 generations.
> 
> In Task 8, we enhanced the fitness function to support **Custom User Preferences**. When a user selects a Strength and Muscular Endurance focus, non-matching activities receive a proportional penalty. The algorithm evolved routines with **100% adherence** to the target goals while strictly respecting safe caloric limits."

---

### Slide 9: Conclusion & Summary
**Visual Elements:**
- Key Takeaways Bullet Points:
  - Automated, evidence-based exercise session design in <0.1 seconds.
  - Multi-attribute continuous fitness function outperforming discrete baselines.
  - Crossover proves essential for multi-domain holistic coverage.
  - Deterministic Crowding & Bespoke Operators ensure diversity and efficiency.
- Final slide text: *Thank you. I welcome your questions.*

**Spoken Commentary (7:30 – 8:00):**
> "To conclude, ACTIVEA successfully bridges evolutionary computation and clinical sports science. By transforming basic discrete rules into a continuous multi-attribute fitness function, we demonstrated that genetic algorithms can generate personalized, balanced, and physiologically safe exercise sessions in under a tenth of a second.
> 
> Thank you for your time. I am now delighted to answer any questions."

---

## 5-Minute Q&A Defense Guide (Tutor Questions & Model Answers)

### Question 1: *"Why did you use a linear tuple chromosome representation instead of a fixed binary string or permutation?"*
**Answer:**  
> "A binary representation is unnatural for this problem because an exercise session involves both categorical choices (which activity to do) and continuous real-valued durations. A permutation representation assumes all activities must be visited once (like TSP), which is inappropriate because a workout should only select 2 to 4 activities out of 10. The linear tuple structure `[(activity, duration)]` directly captures both variables and easily supports gene deduplication, insertion, and deletion."

### Question 2: *"Why did Simulated Annealing achieve a slightly higher fitness score than your GA in Task 4?"*
**Answer:**  
> "Simulated Annealing is a local trajectory method that excels at continuous numerical fine-tuning of existing parameters. It nudged individual duration minutes to land at exactly 449.9 calories. However, GA produced a superior overall workout: ACTIVEA achieved full 100% coverage across all 7 fitness domains, whereas SA was trapped in a local optimum covering only 6 domains. In sports science, domain balance is more critical than a fraction-of-a-calorie difference."

### Question 3: *"How does Deterministic Crowding prevent premature convergence?"*
**Answer:**  
> "In a standard GA, a super-fit individual quickly dominates tournament selection, homogenizing the population. Deterministic Crowding pairs children with their most phenotypically similar parent based on Euclidean distance in exercise allocations. Because a child only replaces its own parent if it is superior, distinct topological niches—such as a yoga-heavy session and a running-heavy session—can coexist without one wiping out the other."

### Question 4: *"How did you utilize the empirical wearable sensor dataset?"*
**Answer:**  
> "The wearable dataset contains 10,000 real sensor records. We split it into an 80% training set and a 20% test partition to measure actual empirical heart rate and Karvonen intensity across activities like walking and running. We used these empirical distributions to validate that our model's caloric expenditure and intensity equations reflect realistic human metabolic responses rather than ungrounded theoretical estimates."
