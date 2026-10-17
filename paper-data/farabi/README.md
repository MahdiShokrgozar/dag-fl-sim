# Result tables of the FARABI paper

One row per run (or per run and node, or per run and reporting time) for the 36
experiment plans behind the paper, 1424 runs in total. `reproduce_paper.py` recomputes
every number of the paper from these tables and writes `paper_numbers.md` / `.csv`, which
also name the plans and runs behind each number.

```bash
python reproduce_paper.py      # Python 3.9+, pandas, scipy
```

The tables were built from the raw run folders (configuration, logs and metric files, a few
GB); the raw folders are not included.

## Files

| File | One row per | Use it for |
|---|---|---|
| `plans.csv` | plan | what each plan runs, where the paper uses it, platform, runs found |
| `runs.csv` | run | configuration and final accuracy of every run: **start here** |
| `timeseries.csv` | run × reporting time (every 30 s) | learning curves |
| `nodes_final.csv` | run × node, at the end | worst honest node, reputation, exclusion, class coverage |
| `summary_final.csv` | run | every column of the simulator's `metrics_summary.csv` at the end |
| `merge_stats.csv` | run | cost per merge, parent composition, trainer-guard firings, FARABI merge weights (parsed from the log) |
| `reproduce_paper.py` | | the selection rules behind each table of the paper, as code |
| `paper_numbers.md` / `.csv` | table / cell | its output |

`run_id` = `<plan>/<experiment>/<arm>/<seed>` joins all files. The plan file of a plan is
`experiments/plans/<plan_file>` (column of `plans.csv`); `run_config.yaml` of a run, from
which the configuration columns were read, is that plan's configuration merged over
`configs/defaults.yaml`.

## Columns of `runs.csv`

| Column | Meaning |
|---|---|
| `run_id`, `plan`, `experiment`, `arm`, `seed` | key; experiment = scenario inside the plan; arm = variation tag, what a plan compares |
| `dataset`, `model`, `n_nodes` | `mnist` / `cifar10`; model; number of nodes |
| `dirichlet_alpha`, `iid` | label skew of the Dirichlet partition (0.5 moderate, 0.1 severe) |
| `frac_poisoning`, `frac_lazy`, `frac_flipping`, `frac_stealing` | share of nodes of each attacker type |
| `attack_mode` | poisoning attack: `gaussian` (received model + N(0, noise_std²) per parameter), `alie`, `ipm`; empty without poisoners |
| `noise_std`, `alie_z`, `ipm_eps` | attack parameters (empty = default / not applicable) |
| `oracle_true_acc` | whether an attacker's block records its true accuracy (default false = 0.0); changes the run, see rule 5 |
| `gossip_k`, `num_tips` | peers per node; maximum parents per merge |
| `backend`, `protocol` | `l2_scaling` (DAG) or `centralized`; `farabi`, `ironforge`, ... |
| `aggregator` | `farabi_global_prox` = FARABI (self-anchored), `farabi_prox` = reputation-anchored variant, `farabi_robust` = the ledger's default "distance gates", `balance`, `rfa`, `lightyear`, `wfagg`, `multi_krum`, `bulyan`, `clippedgossip` |
| `proximal_mu`, `lambda_select`, `lambda_aggregate`, `staleness_enabled` | FARABI distance factor μ; age factor in tip selection and in aggregation |
| `prefilter`, `reputation_weight`, `zero_weight_fallback` | FARABI switches (empty `prefilter` = strategy default) |
| `lr`, `momentum`, `max_batches`, `epochs` | local training |
| `max_time`, `final_clock` | configured and reached simulated time (s) |
| `complete` | **true = usable**: not crashed and finished |
| `has_run_done`, `crashed` | raw completeness markers |
| `final_honest_acc` | **the paper's metric**: mean over honest nodes of each node's accuracy on the full test set at the end (%) |
| `final_honest_acc_min` | worst honest node at the end |
| `final_malicious_acc`, `final_all_nodes_acc`, `n_honest` | same for attacker nodes / all nodes; number of honest nodes |
| `slashed_malicious`, `slashed_honest` | attackers / honest nodes excluded by the ledger by the end |
| `commit` | development commit the run was made on (see the top-level README) |
| `gpu`, `torch`, `python`, `cpu_count`, `pythonhashseed`, `platform_note` | run environment, where recorded |

`timeseries.csv`: `run_id`, `clock` (simulated s), `honest_acc_mean`, `honest_acc_min`,
`honest_acc_max`, `n_honest`, `malicious_acc_mean`, `all_nodes_acc_mean`.
`nodes_final.csv`: `node_type`, `held_out_acc` (full test set), `local_acc` (the node's own
test split), `reputation_score`, `is_slashed`, `class_coverage`, `model_norm`,
`total_traffic_bytes`. `merge_stats.csv`: `merges`, `merges_honest`, `agg_ms_mean` /
`agg_ms_median` (wall time of one merge, ms; compare within a plan only),
`parent_composition_honest` (JSON, `"<parents>,<from attackers>": count`),
`frac_merges_poisoned_majority`, `frac_merges_all_poisoned`, `guard_fired` (the trainer
refused an aggregate and trained from its own model), `zero_weight_merges`, `neff_mean_k0/1/2`
(effective number of parents with 0/1/2 attacker parents), `dist_honest_p10/p50/p90`,
`dist_poisoned_p50`, `weight_poisoned_mean` (FARABI diagnostics, from plan 61 on).

## Rules for using the numbers

1. **Use only `complete` runs.** Five runs are incomplete: plan 59 moderate seed 42 (crashed,
   replaced by 59r) and four LIGHTYEAR cells of 60a, 60b and 60c downloaded before they
   finished (replaced by 60abr and 60cr).
2. **Compare arms only within one plan.** The same configuration on two platforms differs by
   about as much as two seeds do. Within one platform runs are byte-reproducible. The poolings
   used by the paper are those in `reproduce_paper.py`: 58a + 58b, 57b1 + 57b2, 63b1 + 63b2,
   and re-run plans (60abr, 60cr, 59r) that only fill seeds missing from their main plan.
3. Plans 45a and 54 did not fix `PYTHONHASHSEED`: valid, but not byte-reproducible, and the
   two arms of one seed of plan 54 also differ in iteration order.
4. `oracle_true_acc` changes a run: attacker nodes then evaluate their model, which changes the
   random stream. Compare attack runs only with the same setting (plans 62, 64, 66, 67a mu_alie
   and 68 alie_z1 set it).
5. With three parents Multi-Krum falls back to the plain mean (it needs four models) and
   Bulyan to the coordinate-wise median (it needs n ≥ 4f + 3).
6. The trainer's divergence guard runs in every arm; it is part of the simulator.
7. IronForge runs (58d, 59, 59r) use IronForge's own training settings (full epoch, lr 0.05,
   no momentum, gradient clipping 0.2) and an oracle proof of learning.
8. The metric is the honest-node mean on the full test set (10,000 images); SDs in the paper
   are sample SDs over seeds; paired tests are two-sided paired t-tests over seeds.
9. **Exclusion layer.** All plans run the ledger's baseline validation (no exclusion; nobody is
   ever excluded and all proposers keep the same reputation) except 65 and 66, which switch the
   companion paper's exclusion layer on, and 45a, which uses the companion paper's configuration.
