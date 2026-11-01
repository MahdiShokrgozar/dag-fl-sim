# DAG-FL-Sim
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23642993.svg)](https://doi.org/10.5281/zenodo.23642993)

**An Asynchronous Discrete-Event Simulator for Decentralized Federated Learning (DFL) over DAG Ledgers**

DAG-FL-Sim is a high-fidelity **Discrete-Event Simulator (DES)** tailored for researching **Decentralized Federated Learning (DFL)** and **Blockchain-based FL**. Unlike traditional round-based simulators, DAG-FL-Sim models real-world conditions: continuous time, asynchronous communication, network jitter, and heterogeneous hardware capabilities.

At its core, it leverages a **Directed Acyclic Graph (DAG) ledger** for consensus, eliminating single points of failure. Nodes train locally, publish model blocks, choose parent blocks (tips), and merge them asynchronously—each operating on its own clock. 

**Key Features for Researchers:**
- 🛡️ **Byzantine Robustness & Security:** Built-in implementations of advanced poisoning, label-flipping, and backdoor attacks, alongside defense aggregation strategies like FARABI, Krum, RFA, and Bulyan.
- ⚡ **Asynchronous DES Engine:** Accurate modeling of network latency and "straggler" node dynamics.
- 🧱 **DAG Consensus & L2 Scaling:** Pluggable tip selection, block validation, and exclusion layers (including Layer-2 storage offloading integrations).
- 🔌 **Highly Modular:** The aggregation rule, tip selection, reputation system, and attacker behaviors are pluggable strategies defined via YAML configurations. Change one variable and keep the rest strictly controlled.

This is release **1.0.1**. It contains the code, the experiment plans and the result
tables of the paper

> M. Shokrgozar and N. Yazdani. FARABI: A Self-Anchored Aggregation Rule for Non-IID Federated Learning over DAG-Structured Ledgers. Manuscript under review.

## Contents

| Path | What it is |
|---|---|
| `src/` | the simulator: engines, nodes and attackers, strategies (aggregation, selection, consensus, reputation, settlement), metrics |
| `configs/defaults.yaml` | every configuration default; plans override it |
| `experiments/plans/` | the 36 experiment plans behind the paper (table below) |
| `experiments/quickstart.yaml` | a two-arm, one-minute example |
| `paper-data/farabi/` | per-run result tables of those plans (1424 runs) and the script that recomputes every number in the paper |
| `tests/` | unit and integration tests |
| `tools/probes/` | instrumentation that attaches to unchanged simulator code through `sitecustomize` (put the folder on `PYTHONPATH`): parent timing, lazy-gate distances |
| `docs/DEVELOPING.md` | how a run is assembled; adding an aggregation rule, an attacker or a validation variant |

## Installation

Python 3.10 or newer (the paper's runs used 3.12 and 3.13 with PyTorch 2.10 and 2.11).

```bash
pip install -r requirements.txt
```

A GPU is optional; runs fall back to the CPU. MNIST and CIFAR-10 are downloaded by
torchvision into `data/` on first use.

## Quick start

```bash
python run_experiments.py --plan experiments/quickstart.yaml
```

This runs FARABI and RFA for 60 simulated seconds on 10 nodes with severe label skew and
two poisoning nodes (a few minutes on a laptop CPU). Each run writes a folder
`results/quickstart/<experiment>/<arm>/seed_42/` with the merged configuration
(`run_config.yaml`), the metric time series (`metrics_summary.csv`,
`metrics_nodes_detailed.csv`), the run environment (`run_env.txt`) and the log. On a laptop
CPU the two runs took 3.5 minutes and 1.3 GB of memory, and at 60 s the mean node accuracy
(`global_mean_accuracy` in `metrics_summary.csv`) was 83% for FARABI and 21% for RFA;
other platforms give somewhat different numbers.

A plan has a `base_config` per experiment and a list of `variations` (arms), each a partial
configuration merged over `configs/defaults.yaml`. `--jobs N` runs N runs in parallel, each
in its own process. To extend the simulator, see `docs/DEVELOPING.md`.

## Reproducing the paper

Every number in the paper can be recomputed from the result tables, without running the
simulator:

```bash
cd paper-data/farabi
python reproduce_paper.py        # needs pandas and scipy; writes paper_numbers.md / .csv
```

To re-run an experiment, pass its plan to the runner, e.g.

```bash
python run_experiments.py --plan experiments/plans/60c_q1_final_severe_poison.yaml --jobs 2
```

A 10-node run of 600 simulated seconds takes a few minutes on a GPU. Every block keeps its
model, so memory grows with network size: about 9 GB per run at 25 nodes and 18 GB at 50.

| Plan | What it runs | Used in the paper | Platform |
|---|---|---|---|
| [`60a`](experiments/plans/60a_q1_final_moderate.yaml) | Final benchmark, moderate label skew (alpha 0.5), no attackers, 9 aggregators, 5 seeds. | Table main (Moderate) | molab |
| [`60b`](experiments/plans/60b_q1_final_severe.yaml) | Final benchmark, severe label skew (alpha 0.1), no attackers, 9 aggregators, 5 seeds. | Table main (Severe) | molab |
| [`60c`](experiments/plans/60c_q1_final_severe_poison.yaml) | Final benchmark, severe skew with 20% Gaussian poisoning, 9 aggregators, 5 seeds. | Table main (Severe + poisoning), Fig. final and poison-over-time, merge census, Suppl. Table S1 (merge cost) | molab |
| [`60d`](experiments/plans/60d_q1_final_wide_view.yaml) | Final benchmark, severe skew with 20% poisoning, wide view (gossip_k 9, num_tips 9), 9 aggregators, 5 seeds. | Table main (wide view), Suppl. Table S1 (merge cost) | molab |
| [`60abr`](experiments/plans/60abr_q1_final_lightyear_rerun.yaml) | Re-run of LIGHTYEAR cells of 60a/60b that were downloaded before they finished (plus one reproducibility control). | Table main (fills LIGHTYEAR cells) | molab |
| [`60cr`](experiments/plans/60cr_q1_final_severe_poison_rerun.yaml) | Re-run of the LIGHTYEAR seed-99 cell of 60c that was downloaded before it finished. | Table main (fills LIGHTYEAR seed 99) | molab |
| [`68`](experiments/plans/68_clippedgossip.yaml) | ClippedGossip in the four scenarios of Table main and under ALIE z=1, with a FARABI reference arm (byte-identical to 60a-d and 64). | Table main (ClippedGossip row), Sec. attacks closer to the honest band | molab |
| [`61`](experiments/plans/61_reputation_ablation.yaml) | Reputation factor on/off in FARABI, with per-merge weight and distance diagnostics, 5 seeds. | Sec. inside the merge (reputation factor, merge weights) | molab |
| [`56`](experiments/plans/56_prefilter_ablation.yaml) | Weight-space pre-filter ablation: FARABI (both anchors), BALANCE and RFA with and without the filter, 5 seeds. | Table pre-filter, Suppl. Fig. S2 | Kaggle |
| [`57a`](experiments/plans/57a_staleness_scale_10_25.yaml) | Age factor in aggregation on/off at 10 and 25 nodes, severe skew with 20% poisoning, 5 seeds. | Sec. age factor, Sec. network size | Kaggle |
| [`57b1`](experiments/plans/57b1_staleness_scale_50.yaml) | Age factor in aggregation on/off at 50 nodes, seeds 42/7/44. | Sec. age factor, Sec. network size | not recorded |
| [`57b2`](experiments/plans/57b2_staleness_scale_50.yaml) | Age factor in aggregation on/off at 50 nodes, seeds 13/99. | Sec. age factor, Sec. network size | not recorded |
| [`45a`](experiments/plans/45a_staleness_decay_p13.yaml) | Age decay in tip selection and aggregation, lambda choice, 12-node workload of the companion paper, 3 seeds. | Sec. age factor (tip-selection age factor on the companion paper's workload) | Kaggle |
| [`67a`](experiments/plans/67a_mu_sweep_lambda_select.yaml) | Sweep of mu (0.001 to 4) under Gaussian poisoning and (0.5, 2, 4) under ALIE z=1; tip-selection age factor on/off. | Discussion (calibration of mu), Sec. age factor, Sec. attacks closer to the honest band | molab |
| [`63a`](experiments/plans/63a_baselines_at_scale_25.yaml) | FARABI and four baselines at 25 nodes (5 poisoners), severe skew, 5 seeds. | Table scale (25 nodes) | molab |
| [`63b1`](experiments/plans/63b1_baselines_at_scale_50_s1.yaml) | FARABI and four baselines at 50 nodes (10 poisoners), seeds 42/7/44. | Table scale (50 nodes) | molab |
| [`63b2`](experiments/plans/63b2_baselines_at_scale_50_s2.yaml) | FARABI and four baselines at 50 nodes (10 poisoners), seeds 13/99. | Table scale (50 nodes) | molab |
| [`62`](experiments/plans/62_noise_scale_sweep.yaml) | Gaussian poisoning with sigma 10, 0.1, 0.01 and 0.003 per parameter, five aggregators, 5 seeds. | Sec. attacks closer to the honest band (noise sweep) | molab |
| [`64`](experiments/plans/64_in_band_attacks.yaml) | ALIE (z = 0.25 and 1) and model-space IPM (eps 0.1 and 1), five aggregators, 5 seeds. | Table ALIE (structured in-band attacks) | molab |
| [`65`](experiments/plans/65_full_ledger_t2rc.yaml) | Severe-skew scenario with the companion paper's exclusion layer on (distance gates, one strike): 7 aggregators under 20% poisoning and an honest control, 5 seeds. | Sec. inside the merge (with the exclusion layer on) | molab |
| [`66`](experiments/plans/66_alie_full_ledger.yaml) | ALIE z=1 with the exclusion layer on. The 'legacy' arms use an accuracy detector whose evidence the voting nodes cannot recompute, so by design it excludes nobody; they are not used in the paper. | Discussion (ALIE and the exclusion layer); only the distance-gates arm is used | molab |
| [`58a`](experiments/plans/58a_cifar_severe_poison_s1.yaml) | CIFAR-10, severe skew with 20% poisoning, 6 aggregators, seeds 42/7/44. | Table CIFAR-10 | Kaggle |
| [`58b`](experiments/plans/58b_cifar_severe_poison_s2.yaml) | CIFAR-10, severe skew with 20% poisoning, 6 aggregators, seeds 13/99. | Table CIFAR-10 | Kaggle |
| [`58c`](experiments/plans/58c_cifar_references.yaml) | CIFAR-10 at alpha 0.5 without attackers (FARABI, distance gates, centralized FedAvg) and a centralized reference at alpha 0.1, 5 seeds. | Sec. CIFAR-10 (moderate skew, centralized references) | molab |
| [`58d`](experiments/plans/58d_cifar_ironforge.yaml) | IronForge protocol on CIFAR-10, severe skew with 20% poisoning, 5 seeds. | Table CIFAR-10 (IronForge row) | Kaggle |
| [`55`](experiments/plans/55_cifar10_calibration_q1.yaml) | CIFAR-10 calibration with a centralized FedAvg reference, one seed. | Sec. CIFAR-10 (one-seed calibration) | Kaggle |
| [`67b`](experiments/plans/67b_view_width_separation.yaml) | The four corners of gossip_k in {6, 9} x num_tips in {3, 9} for FARABI, RFA and the distance gates, severe skew with 20% poisoning, 5 seeds; corners (6,3) and (9,9) reproduce 60c and 60d exactly. | Sec. main benchmark (wide view: number of peers vs number of parents) | molab |
| [`69`](experiments/plans/69_ironforge_control.yaml) | IronForge and FARABI, each with IronForge's training settings and with ours, in the four scenarios of Table main, 5 seeds; the FARABI/our-training arm reproduces 60a-d exactly. | Sec. protocol-level comparison with IronForge; Suppl. Tables S2 and S3 | molab |
| [`70`](experiments/plans/70_kernel_ablation.yaml) | FARABI with its exponential kernel replaced by a hard radius (2, 4, 50) around the node's own model, severe skew with and without 20% poisoning and two small-noise attacks, 5 seeds; the kernel arms reproduce 60c, 60b and 62 exactly. | Sec. inside the merge (kernel vs hard radius), Suppl. Table S6 | molab |
| [`71`](experiments/plans/71_baseline_sensitivity.yaml) | Hyperparameter sensitivity of BALANCE (gamma, alpha) and LIGHTYEAR (tau), severe skew with and without 20% poisoning, 5 seeds; the published settings reproduce 60c and 60b exactly. | Discussion (calibration of the baselines), Suppl. Table S5 | molab |
| [`72`](experiments/plans/72_age_factor_interaction.yaml) | Age factor in aggregation x age factor in tip selection (2x2), FARABI, severe skew with and without 20% poisoning, 5 seeds. | Sec. age factor | molab |
| [`73`](experiments/plans/73_headline_seed_extension.yaml) | Five more seeds for the aggregators that keep learning in the headline scenario of Table main (severe skew with 20% poisoning). | Sec. seeds and statistics; Suppl. Table S9 | molab |
| [`76a`](experiments/plans/76a_entry_reputation.yaml) | Entry reputation (constant, stake-derived, attackers at minimum stake) x reward rate, self- and reputation-anchored FARABI, severe skew with and without 20% poisoning, 5 seeds. | Sec. choice of anchor; Suppl. Table S7 | molab |
| [`76b`](experiments/plans/76b_entry_reputation.yaml) | As 76a under ALIE with z = 1. | Sec. choice of anchor; Suppl. Table S7 | molab |
| [`78`](experiments/plans/78_wfagg_sensitivity.yaml) | WFAgg with varied attacker count, mixing weight and filter rule, severe skew with and without 20% poisoning, 5 seeds. | Discussion (calibration of the baselines), Suppl. Table S8 | molab |
| [`79`](experiments/plans/79_main_table_seed_extension.yaml) | Five more seeds for the aggregators that keep learning in the other three scenarios of Table main. | Sec. seeds and statistics; Suppl. Table S9 | molab |

**Reproducibility.** The runner fixes `PYTHONHASHSEED`, so on one hardware and software
platform two runs with the same seed are identical to the byte. Across platforms (GPU model,
PyTorch version) the order of floating-point reductions changes and the training dynamics
amplify it to differences of the size of the seed spread; compare arms only within one plan.
This release is a snapshot of development commit `a6177bc`; each run in
`paper-data/farabi/runs.csv` records the development commit it ran on, which for most
plans is earlier. Reference arms of plans 67a, 67b, 68 and 69 re-ran, on this snapshot,
the FARABI arm of the final benchmark (plans 60a to 60d) and of the ALIE scenario (plan
64) and the RFA and distance-gate arms of plans 60c and 60d: all 85 of those runs reproduced
the earlier ones exactly. The other arms were not re-run on it. The platform of each plan is in the table above.

**About the code comments.** Many comments cite dated entries of the authors' design log,
which is not part of this release, and some comments and log messages are in Persian. They do
not affect behavior; the source files are exactly those the experiments ran on.

## License and citation

MIT license (see `LICENSE`). To cite the software, see `CITATION.cff`; to cite the method,
cite the paper above.
