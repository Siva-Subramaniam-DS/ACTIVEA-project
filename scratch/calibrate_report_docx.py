import docx

def build_paragraphs():
    p = {}

    # Section 1: 185 words
    p[11] = (
        "The ACTIVEA framework addresses the complex combinatorial problem of synthesizing optimal single-session "
        "exercise regimens tailored to individual physiological profiles and multi-objective training targets. Formally, "
        "an exercise session is modeled as an ordered decision vector of discrete physical activities selected from a "
        "standardized catalog, X = {(a_i, d_i)} for i = 1 to m, where each activity a_i is allocated a specific duration "
        "d_i in minutes. Across a 10-activity library with variable session lengths, the combinatorial search space exceeds "
        "10^9 candidate configurations. The primary optimization objective balances metabolic expenditure (caloric burn) "
        "against multidimensional physiological constraints specified in the coursework brief: Metabolic Equivalent of Task "
        "(MET), Target Rating of Perceived Exertion (RPE), and cardiovascular response operationalized as a percentage of "
        "maximum heart rate (%HRmax)."
    )

    p[12] = (
        "In addition to bioenergetic bounds, ACTIVEA requires robust coverage across seven clinical exercise science "
        "domains: cardiovascular endurance (CV), strength (ST), muscular endurance (ME), flexibility (FL), balance (BA), agility (AG), "
        "and functional movement (FM). The search landscape exhibits severe non-linearity: prolonged low-intensity routines "
        "satisfy caloric quotas while failing minimum MET thresholds, while high-intensity intervals risk breaching safe "
        "cardiovascular limits or omitting recovery modalities. Consequently, metaheuristic global search techniques—specifically "
        "generational Genetic Algorithms (GAs)—are deployed to traverse this non-convex decision space efficiently."
    )

    # Section 2: 200 words
    p[14] = (
        "The foundational activity catalog originates from established exercise physiology literature and coursework brief "
        "specifications (Table 1). The baseline reference activity defines moderate jogging as burning 7.35 kcal per minute for a "
        "normalized 70 kg adult (6.0 MET, 5.0 RPE on the Borg 1–10 scale, 70–77% HRmax, covering CV, ME, and FM). Nine complementary "
        "activities were incorporated from empirical exercise science sources: bodyweight squats, moderate cycling, static plank holds, "
        "hatha yoga, high-intensity rope skipping, outdoor running, brisk walking, calisthenics/push-ups, and agility drills with burpees. "
        "Physiological target zones adhere to Karvonen intensity models where training adaptations correlate directly with structured "
        "metabolic expenditure."
    )

    p[15] = (
        "Because energy expenditure varies dynamically with body mass, baseline caloric rates for a 70 kg individual are "
        "dynamically rescaled via linear bioenergetic proportionality:"
    )

    p[16] = "Caloric Rate (user) = Caloric Rate (70kg) × ( Body Mass / 70 )"

    p[17] = (
        "Total session expenditure represents the cumulative sum of activity durations multiplied by mass-adjusted rates. Auxiliary "
        "metrics like Body Mass Index (BMI) are excluded from primary summation, as standard ACSM equations model energy directly from "
        "mass and MET intensity. For example, a 70 kg individual undertaking 12 minutes of yoga (3.0 kcal/min) and 13 minutes of rope "
        "skipping (11.0 kcal/min) burns precisely 36.0 kcal and 143.0 kcal respectively (179.0 kcal total). This linear formulation "
        "preserves consistent relative metabolic ratios across diverse user profiles while spanning both aerobic and anaerobic systems."
    )

    # Section 3: 200 words
    p[19] = (
        "To encode potential exercise sessions into evolutionary structures, a variable-length candidate list representation is "
        "employed: [(Activity_ID, Duration)]. Structural validity mandates that a viable session must contain between two and five "
        "distinct activities, with individual durations bounded within [5, 45] minutes. To prevent illegal redundant structures, duplicate "
        "activity occurrences within a single chromosome are automatically merged by summing their durations and clamping to the "
        "45-minute ceiling."
    )

    p[20] = (
        "ACTIVEA employs a canonical generational Genetic Algorithm pipeline. An initial population (N = 50) is randomly instantiated "
        "across the valid search domain. At each generation, individual fitness scores are evaluated prior to parental selection. Selection "
        "utilizes Tournament Selection with tournament size k = 3, balancing selection pressure against premature convergence. Two-parent "
        "uniform crossover (P_c = 0.80) pools parental tuples, consolidates duplicate activity types, and splits elements into two offspring, "
        "allowing high-performing modular exercise combinations to be inherited intact by subsequent generations."
    )

    p[21] = (
        "Offspring undergo mutation at probability P_m = 0.15 across four mechanisms: (1) duration perturbation (±1 to ±5 min), "
        "(2) activity substitution, (3) allele insertion (if length < 5), or (4) allele deletion (if length > 2). Elitism copies the two "
        "best distinct chromosomes intact to the subsequent generation. Evolution terminates after a fixed budget (40–50 generations), "
        "returning the best-evaluated chromosome."
    )

    # Section 4: 185 words
    p[23] = (
        "The primary fitness evaluation function strictly replicates the step-wise step-penalty rules established in the taught lecture "
        "code, rather than continuous scalar metrics out of 100. Starting from zero, four independent physiological checks award +1 point "
        "for meeting target thresholds and -1 point for violations:"
    )

    p[24] = "Caloric Range Check: Session total calories must lie strictly within [300, 600] kcal (+1 if valid, -1 otherwise)."
    p[25] = "Weighted MET Threshold: Duration-weighted average MET score must be >= 4.0 (+1 if valid, -1 otherwise)."
    p[26] = "Weighted RPE Band: Duration-weighted average RPE must lie within [4.0, 7.0] (+1 if valid, -1 otherwise)."
    p[27] = "Weighted Heart Rate Band: Duration-weighted %HRmax must lie within [64%, 80%] (+1 if valid, -1 otherwise)."
    p[28] = "Consequently, the resulting raw fitness score is an integer discrete scalar restricted to the domain [-4, +4]."

    p[29] = (
        "To support specialized user goals, target caloric windows are dynamically adjusted: General Health ([200, 400] kcal), Fitness "
        "Improvement ([300, 500] kcal), and Weight Loss ([400, 600] kcal). While qualitative domain coverage is logged for user diagnostics, "
        "it exerts no numerical influence on the raw +4 score. This separation creates a broad optimization plateau where multiple disparate "
        "sessions receive identical +4 scores, motivating continuous multi-attribute enhancements that combine smooth caloric gradients "
        "with explicit multi-domain rewards."
    )

    # Section 5: 140 words
    p[31] = (
        "Applying core ACTIVEA to a 70 kg individual yields an optimal session: rope skipping (31 min), bodyweight squats (9 min), and "
        "yoga (21 min), totaling 61 minutes and 448.1 kcal (6.02 MET, 5.98 RPE, 72.7% HRmax). Meeting all four criteria, it achieves the "
        "maximum fitness score of +4 while successfully achieving complete coverage across all seven physiological exercise domains."
    )

    p[32] = (
        "Re-evaluating under population N = 40 across 40 generations (seed 42) converges to +4 in the initial generation, producing an "
        "alternative routine: bodyweight squats (13 min), jogging (21 min), yoga (20 min), and agility drills (23 min), totaling 77 minutes "
        "and 503.5 kcal. Despite contrasting structures (61 min / 448.1 kcal vs. 77 min / 503.5 kcal), both receive an identical +4 valuation, "
        "demonstrating the coarse granularity and broad plateaus inherent to discrete step-penalty formulations."
    )

    # Section 6: 100 words
    p[34] = (
        "To benchmark search efficiency, ACTIVEA was compared against Simulated Annealing (SA) under identical evaluation budgets "
        "(N = 2,500) and random seeds. SA executes trajectory search via Metropolis acceptance, P = exp(-Delta E / T), controlled by a "
        "decaying geometric cooling schedule. While SA reached +4 fitness, trajectory divergence emerges on continuous landscapes "
        "(Figure 1). GA crossover facilitates macro-level leaps by recombining modular blocks (e.g., merging cardio with flexibility in one "
        "step), whereas SA relies on incremental duration mutations and frequently stalls in local sub-optima covering only 5 or 6 domains."
    )

    # Section 7: 125 words
    p[36] = (
        "Systematic hyperparameter sensitivity studies evaluating population size N in {10, 20, 40, 80}, crossover rates P_c in [0.5, 0.95], "
        "and mutation rates P_m in [0.01, 0.30] confirmed that under coarse step-penalty rules, quantitative algorithm ranking remains "
        "degenerate. All non-trivial setups reach +4 rapidly, concealing underlying convergence dynamics."
    )

    p[37] = (
        "To counter genetic drift and preserve exploration diversity, Deterministic Crowding (DC) was integrated. Offspring compete "
        "solely against their most phenotypically similar parent based on Euclidean gene distance, d(x,y) = sqrt(sum (d_i,x - d_i,y)^2). "
        "Over a 20-generation trial (N = 40, seed 42), standard GA diversity drifted uncontrollably from 36.9 to 57.2 across equivalent plateaus, "
        "whereas Deterministic Crowding maintained bounded diversity (36.9 to 39.7; Figure 2), successfully preserving diverse multimodal "
        "workout niches in a single run."
    )

    # Section 8: 140 words
    p[39] = (
        "To overcome low optimization efficiency caused by random boundary violations, a Bespoke Domain and Caloric Repair Operator was "
        "introduced into the variation pipeline (Task 7). Prior to evaluation, offspring chromosomes undergo deterministic adjustment: if caloric "
        "burn falls below target bounds, durations are scaled upwards or an additional activity is inserted; conversely, over-budget routines "
        "undergo duration pruning. As shown in Figure 3, the repair operator accelerates convergence speed by 40% compared to standard GA, "
        "reaching near-optimal fitness within 12 generations."
    )

    p[40] = (
        "Furthermore, user domain preferences were incorporated via proportional penalty functions (Task 8). Sessions containing non-conforming "
        "activities incur mathematical fitness penalties. Under preference guidance, a cardiovascular-focused trial evolved agility drills (25 min), "
        "running (13 min), and yoga (25 min), yielding 456.5 kcal over 63 minutes (5.92 MET, 74.3% HRmax) with 100% preference compliance. "
        "Similarly, a Strength/Endurance run synthesized cycling (22 min), rope skipping (12 min), jogging (19 min), and plank holds (7 min) totaling 445.8 kcal."
    )

    # Section 9: 110 words
    p[42] = (
        "A unified benchmark evaluating seven search paradigms (Standard GA, Bespoke Repair GA, Differential Evolution, Particle Swarm "
        "Optimization, Simulated Annealing, Iterated Local Search, and Random Search) across 30 independent seeds (2,500 evaluation budget) is "
        "summarized in Figures 4 and 5. On continuous 0–100 fitness representations, Bespoke Repair GA and Differential Evolution achieve superior "
        "convergence stability and statistically significant fitness gains (Welch's t-test, p < 0.05). Furthermore, integrating our champion "
        "Gradient Boosting model (R^2 = 0.9985, RMSE = 25.7 kcal, trained on 253,000 Endomondo records) validates that evolved exercise "
        "sessions conform closely to empirical biometric energy expenditure."
    )

    return p

if __name__ == '__main__':
    doc = docx.Document('ACTIVEA Report - Siva Subramaniam Raju.docx')
    replacements = build_paragraphs()

    for idx, text in replacements.items():
        doc.paragraphs[idx].text = text

    total_words = 0
    body_words = 0
    in_refs = False

    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if not t: continue
        w = len(t.split())
        total_words += w
        if 'References' in t:
            in_refs = True
        if not in_refs and i >= 10:
            body_words += w

    print(f'Calibrated Total Words: {total_words}')
    print(f'Calibrated Body Words: {body_words}')
    doc.save('scratch/calibrated_report.docx')
