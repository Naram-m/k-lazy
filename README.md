# Partially-Lazy Gradient Descent (k-lazyGD)

This repository contains the implementation accompanying the paper:

> **Partially-Lazy Gradient Descent for Smoothed Online Learning**  

The repository provides the code used to reproduce all experiments and figures reported in the paper, including the **illustrative examples**, **shifting stochastic phases experiment**, **corrupted phases experiment**, and **worst-case sequences experiment**.

---

## Overview

This implementation reproduces all empirical results, including:
- Switching and hitting cost visualization for illustrative examples (Figures 1 & 2),
- Shifting stochastic sequence experiment (Appendix Figure 4),
- Corrupted sequence experiment (Appendix Figure 6),
- Lower-bound adversarial construction (Appendix Figure 7).

---

## Repository Structure

```
klazy_src/
│
├── GD.py                 # Greedy Online Gradient Descent
├── LGD.py                # Lazy Gradient Descent (Dual Averaging)
├── Klazy.py              # Partially-Lazy Gradient Descent (k-lazyGD)
│
├── intro/                # Illustrative examples from the introduction
│   ├── ex1_results/      # Stores plotting scripts for actions and resulting figures for example 1
│   ├── ex2_results/      
│   ├── ex1.py            # runs example 1 and saves the results in the corresponding folder
│   └── ex2.py          
│
├── stochastic/           # Shifting stochastic sequences experiment
│   ├── main.py           # Main entry script for this experiment
│   ├── klazy_results_stoch/
│   │   ├── hc_acc.py     # Plots hitting cost
│   │   ├── sc_plot.py    # Plots switching cost
│   │   └── R_T.py        # Plots total regret│
├── corrupted/            # Corrupted phase experiment
│   ├── analogous structure to `stochastic/`
│
├── worst_case/           # Adversarial lower-bound construction
│   ├── analogous structure to `stochastic/`
│
├── sequence_plotting_corr.py   # Generates sequence visualization for corrupted example (Appendix Fig. 5)
└── sequence_plotting_stoch.py  # Generates sequence visualization for stochastic example (Appendix Fig. 3)
```

---

## Setup

This codebase is written in **Python 3.10+** and uses standard numerical libraries.


---

## Running the Experiments

Each experiment can be run independently from its respective folder.  
All experiments use the same three learners:

- `GD` – Greedy Online Gradient Descent  
- `LGD` – Lazy (Dual-Averaging) Gradient Descent  
- `k-LazyGD` – Proposed partially-lazy update with laziness slack \(k\)

### Illustrative Examples (Introduction)
```
python main_intro.py
```
Produces **Figures 1 and 2** showing switching and hitting costs.

**Tip:**  
Set `fixed_sigma = True` when storing the actions (to capture snapshots with arrows).  
Otherwise, use the standard schedule:
\[
\sigma_t = \sqrt{t}.
\]

### Shifting Stochastic Sequences
```
cd stochastic
python main.py
```
Make sure to set the regularization rate \(\sigma\) consistently across learners for each experiment:
```python
# in GD.py, LGD.py, and Klazy.py
self.sigma = np.sqrt(len(self.actions)) / np.sqrt(4 * 15)
```
This reproduces (switching cost, hitting cost, and total regret).

### Corrupted Phase Experiment
```
cd corrupted
python main.py
```
Uses the same structure as the stochastic case with:
```python
self.sigma = np.sqrt(len(self.actions)) / 4
```
This reproduces the corrupted sequences experiment.

### Worst-Case Adversarial Sequence
```
cd worst_case
python main.py
```
Set:
```python
self.sigma = np.sqrt(len(self.actions)) / np.sqrt(10)
```
This reproduces the **lower-bound example** from the appendix.

---

## Configuration Summary

Each agent defines four preset values for the regularization rate `sigma` corresponding to the four experimental setups.
These are also indicated in the code of the learners with comments.

| Experiment Type         | Sigma Definition                                |
|--------------------------|-------------------------------------------------|
| Introductory Example     | `np.sqrt(len(actions))`                         |
| Shifting Stochastic Seq. | `np.sqrt(len(actions)) / np.sqrt(4 * 15)`       |
| Corrupted Phases         | `np.sqrt(len(actions)) / 4`                     |
| Worst Case (Lower Bound) | `np.sqrt(len(actions)) / np.sqrt(10)`           |

---

## Reproducing Figures

To visualize results, run the plotting scripts inside each experiment’s results folder.  
For example:
```
cd stochastic/plotting
python sc_plot.py     # Switching cost
python hc_acc.py      # Hitting cost
python R_T.py         # Total regret
```

The generated figures correspond to those reported in the main paper and supplementary material.

---
