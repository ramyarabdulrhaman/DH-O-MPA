# DH-O-MPA: Trust-Guided Path Repair for SIoT Service Discovery

Code, data, and per-run results for the paper
*"DH-O-MPA: Trust-Guided Path Repair and Biased Random-Key Decoding in a Multi-Objective Marine Predators Algorithm for Social IoT Service Discovery"*.

DH-O-MPA searches for multi-hop service paths in a Social IoT network. Each path is scored on four objectives:
- trust (geometric mean of link trust);
- end-to-end latency;
- energy;
- QoS bottleneck.

Every path must also satisfy three constraints: a deadline, a minimum relay battery, and a minimum relay trust.

The algorithm has three components:
1. **Biased random-key decoder.** It builds a simple path hop by hop from position keys, plus a trust and latency prior.
2. **Trust-guided Path Repair (TPR).** It removes the weakest relay, reroutes around it, and writes the repair back into the keys.
3. **Diversity-driven adaptive FADs.**

## Repository layout

| Path | Content |
|---|---|
| `scripts/experiment.py` | Instance generators, shared decoder, DH-O-MPA and all baselines (MPA, NSGA-II, NSGA-III, MOEA/D, MOPSO, ZOA, CPO, WS-KSP), experiment runner |
| `scripts/analyze.py` | Hypervolume, IGD, Wilcoxon + Holm, Vargha–Delaney A12, Friedman test |
| `scripts/make_tables.py` | LaTeX tables and number macros generated from the result files |
| `scripts/make_figures.py` | All result figures |
| `data/loc-brightkite_edges.txt.gz` | SNAP Brightkite friendship graph ([source](https://snap.stanford.edu/data/loc-brightkite.html)) |
| `results/raw_{tune,test}.pkl` | Raw logs of every run (final archives, paths, convergence snapshots) |
| `results/per_run_*.csv` | Metrics for each run |
| `results/summary_*.csv` | Mean and std per algorithm and instance |
| `results/stats_test.csv` | Pairwise tests vs. DH-O-MPA (raw and Holm p-values, A12, outcome) |
| `results/friedman_test.csv` | Friedman mean ranks |
| `results/instances_*.csv` | Instance metadata (size, endpoints, hop budget, deadline, malicious relays) |
| `figures/` | Result figures used in the paper |

## Protocol

- **Tuning** instances (used only for design decisions): `rgg300@1001`, `rgg800@1002`, `hospital@1003`, `ba500@1004`.
- **Test** instances (run once with the frozen configuration `FINAL_CONFIG`): `rgg500@42`, `rgg1000@42`, `rgg2000@42`, `hospital@42`, `brightkite@42`, `ba500@42`, `ba1500@42`.
- **Budget:** 3,000 evaluations per run, 30 independent runs, population 30, archive 40. WS-KSP is deterministic and runs once.

All instances are generated deterministically from the seed after `@`.

## Reproduce

Requires Python 3.12. Then:

```bash
pip install -r requirements.txt
cd scripts
python experiment.py --phase test --workers 14   # about 11 min on a 16-thread CPU; writes results/raw_test.pkl
python analyze.py test                           # metrics and statistics -> results/*.csv
python make_tables.py                            # -> tables/*.tex
python make_figures.py                           # -> figures/*.png
```

Use `--phase tune` to reproduce the tuning study. The provided `results/` folder already contains the logs used in the paper, so `analyze.py`, `make_tables.py` and `make_figures.py` can be run without re-running the experiments.

## Main results (test set, hypervolume)

- **Ranking:** DH-O-MPA has the best Friedman rank among the stochastic optimizers.
- **Against the metaheuristics:** it is significantly better than every metaheuristic on 6 of the 7 instances. On BA-500 it ties with MPA.
- **Against WS-KSP:** the deterministic WS-KSP heuristic is better on the four short-route instances (BA-500, BA-1500, Brightkite, Hospital). It finds no feasible path on the long-route RGG networks.
- **RGG-2000:** no method solves this instance reliably.

See `results/summary_test.csv` and `results/stats_test.csv` for the full numbers.

## Notes

- Device attributes, interaction histories and attacks are simulated. Only the Brightkite topology is real data.
- ZOA and CPO are re-implementations adapted to multi-objective search. NSGA-II, NSGA-III and MOEA/D use pymoo.
