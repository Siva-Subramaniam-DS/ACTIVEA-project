# Head-to-Head Algorithm Comparison (Powered by Champion ML Caloric Model)

**Goal Profile:** Fitness Improvement  
**Evaluation Budget:** 2,500 evaluations per run  
**Replications:** 10 independent stochastic trials  
**Hypothesis Testing:** Welch's two-sample unequal-variance t-test against Standard GA  

| Algorithm | Mean Fitness +/- Std | Min Fit | Max Fit | Runtime (s) | p-value (vs GA) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Differential Evolution | 97.74 +/- 0.68 | 96.81 | 98.35 | 10.63s | 0.0005 | Superior (p < 0.05) * |
| Simulated Annealing | 94.69 +/- 1.70 | 92.24 | 98.02 | 9.53s | 0.0394 | Superior (p < 0.05) * |
| Bespoke Repair GA | 93.88 +/- 1.38 | 90.91 | 95.08 | 10.69s | 0.1187 | Equivalent (p >= 0.05) |
| Iterated Local Search | 92.37 +/- 8.02 | 69.77 | 96.98 | 9.54s | 0.8590 | Equivalent (p >= 0.05) |
| Standard Genetic Algorithm | 91.87 +/- 3.51 | 86.00 | 95.98 | 9.52s | Baseline (N/A) | Reference Baseline |
| Particle Swarm Optimization | 91.80 +/- 8.02 | 69.99 | 98.35 | 9.64s | 0.9811 | Equivalent (p >= 0.05) |
| Random Search | 90.94 +/- 2.71 | 87.33 | 94.86 | 9.51s | 0.5150 | Equivalent (p >= 0.05) |

*Note: An asterisk (*) indicates statistically significant difference at alpha = 0.05 level.*