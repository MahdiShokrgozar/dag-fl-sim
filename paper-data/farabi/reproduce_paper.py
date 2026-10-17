#!/usr/bin/env python3
"""Recompute every result number of the Q1 paper from the tidy dataset.

Reads runs.csv and merge_stats.csv (built by build_dataset.py) and writes
  paper_numbers.md   the paper's tables, each with the plans and runs it comes from
  paper_numbers.csv  one row per reported cell (table, row, column, value, n, source)

The selection rules (which plan feeds which table, how re-runs fill truncated
cells) are written out below, one function per table, so a reader can check them
against the paper and reuse them for new analyses.

Usage: python reproduce_paper.py        (needs pandas and scipy)
"""
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
runs = pd.read_csv(HERE / "runs.csv")
merge = pd.read_csv(HERE / "merge_stats.csv") if (HERE / "merge_stats.csv").exists() else None
ok = runs[runs.complete == True]  # noqa: E712  (only complete, non-crashed runs are ever used)

MD, CELLS = [], []


def cells(plans, experiment, arm, fill_from=()):
    """{seed: final honest accuracy} from `plans`; `fill_from` plans only add missing seeds."""
    sel = ok[ok.plan.isin(plans) & (ok.experiment == experiment) & (ok.arm == arm)]
    out = dict(zip(sel.seed, sel.final_honest_acc))
    for p in fill_from:
        extra = ok[(ok.plan == p) & (ok.experiment == experiment) & (ok.arm == arm)]
        for s, v in zip(extra.seed, extra.final_honest_acc):
            out.setdefault(s, v)
    return out


def ms(d):
    v = list(d.values())
    if not v:
        return "n/a", 0
    sd = pd.Series(v).std() if len(v) > 1 else 0.0  # sample SD (n-1)
    return f"{pd.Series(v).mean():.1f} ± {sd:.1f}", len(v)


def paired(a, b):
    """mean(a-b) over common seeds, positives, two-sided paired t-test p."""
    s = sorted(set(a) & set(b))
    if len(s) < 2:
        return "n/a"
    d = [a[k] - b[k] for k in s]
    p = stats.ttest_rel([a[k] for k in s], [b[k] for k in s]).pvalue
    return f"{sum(d)/len(d):+.1f} ({sum(x > 0 for x in d)}/{len(s)}, p={p:.3f})"


def table(title, source, header, rows):
    MD.append(f"\n## {title}\n\n*Source:* {source}\n")
    MD.append("| " + " | ".join(header) + " |")
    MD.append("|" + "---|" * len(header))
    for r in rows:
        MD.append("| " + " | ".join(str(x) for x in r) + " |")
        for h, x in zip(header[1:], r[1:]):
            CELLS.append({"table": title, "row": r[0], "column": h, "value": x, "source": source})


# ------------------------------------------------------------------ Table main (Sec. main benchmark)
ARMS = ["global_prox", "prox", "lightyear", "balance", "wfagg", "bulyan", "multi_krum", "rfa", "robust"]
SCEN = [("Moderate", "60a", "1_non_iid_moderate_honest", ["60abr"]),
        ("Severe", "60b", "2_non_iid_severe_honest", ["60abr"]),
        ("Severe + poisoning", "60c", "3_non_iid_severe_with_poisoning", ["60cr"]),
        ("Severe + poisoning, wide", "60d", "3w_non_iid_severe_with_poisoning_wide", [])]


def t_main():
    rows = []
    for arm in ARMS:
        row = [arm]
        for _, p, e, fill in SCEN:
            v, n = ms(cells([p], e, arm, fill))
            row.append(f"{v} (n={n})")
        rows.append(row)
    # ClippedGossip ran later (plan 68); its global_prox arm is byte-identical to 60a-d (checked below).
    rows.append(["clippedgossip (plan 68)"] + [f"{ms(cells(['68'], e, 'clippedgossip'))[0]} (n={ms(cells(['68'], e, 'clippedgossip'))[1]})"
                                              for _, p, e, fill in SCEN])
    same = all(cells(["68"], e, "global_prox") == cells([p], e, "global_prox") for _, p, e, _ in SCEN)
    table("Table main: final honest-node accuracy, MNIST", "plans 60a-d; truncated LIGHTYEAR cells filled from 60abr "
          "(Moderate s99; Severe s13, s99) and 60cr (Severe+poisoning s99); ClippedGossip from plan 68 "
          f"(its global_prox arm identical to 60a-d in every seed: {same})",
          ["aggregator"] + [s for s, *_ in SCEN], rows)
    rows = []
    for arm in ARMS[1:]:
        row = [f"global_prox − {arm}"]
        for _, p, e, fill in SCEN:
            row.append(paired(cells([p], e, "global_prox", fill), cells([p], e, arm, fill)))
        rows.append(row)
    rows.append(["global_prox − clippedgossip (plan 68)"] + [paired(cells(["68"], e, "global_prox"), cells(["68"], e, "clippedgossip"))
                                                           for _, p, e, fill in SCEN])
    table("Paired differences, main benchmark", "as Table main; paired by seed, two-sided paired t-test",
          ["comparison"] + [s for s, *_ in SCEN], rows)


# ------------------------------------------------------------------ Table prefilter (plan 56)
def t_prefilter():
    e = "3_non_iid_severe_with_poisoning"
    pairs = [("Farabi-Prox, self", "global_prox_nofilter", "global_prox"),
             ("Farabi-Prox, rep.", "prox_nofilter", "prox"),
             ("BALANCE", "balance", "balance_filter"), ("RFA", "rfa", "rfa_filter")]
    rows = []
    for name, off, on in pairs:
        a, b = cells(["56"], e, off), cells(["56"], e, on)
        rows.append([name, ms(a)[0], ms(b)[0], paired(b, a)])
    table("Table prefilter: pre-filter ablation", "plan 56 (Kaggle), severe + 20% poisoning, 5 seeds",
          ["aggregator", "no filter", "filter", "filter − none"], rows)


# ------------------------------------------------------------------ Sec. inside the merge (plan 61)
def t_reputation():
    rows = []
    for name, e in [("Severe, honest", "2_non_iid_severe_honest"), ("Severe + poisoning", "3_non_iid_severe_with_poisoning")]:
        a, b = cells(["61"], e, "global_prox"), cells(["61"], e, "global_prox_norep")
        rows.append([name, ms(a)[0], ms(b)[0], paired(a, b)])
    table("Reputation factor on/off", "plan 61 (molab)", ["scenario", "with reputation", "constant r", "with − without"], rows)
    if merge is not None:
        m = merge.merge(runs[["run_id", "plan", "experiment", "arm"]], on="run_id")
        m = m[(m.plan == "61") & (m.arm == "global_prox")]
        rows = []
        for e, g in m.groupby("experiment"):
            rows.append([e, f"{g.neff_mean_k0.mean():.2f}", f"{g.neff_mean_k1.mean():.2f}" if g.neff_mean_k1.notna().any() else "—",
                         f"{g.dist_honest_p10.mean():.2f}–{g.dist_honest_p90.mean():.2f}",
                         f"{g.dist_poisoned_p50.mean():.0f}" if g.dist_poisoned_p50.notna().any() else "—"])
        table("Merge weights (n_eff, distances)", "plan 61, global_prox arm, per-merge diagnostic log line; "
              "means over 5 runs of per-run statistics (the paper quotes pooled percentiles, which differ slightly)",
              ["scenario", "n_eff, 3 honest parents", "n_eff, 1 poisoned", "honest dist p10–p90", "poisoned dist p50"], rows)


# ------------------------------------------------------------------ Sec. age factor + network size (plans 57a/b)
def t_staleness():
    rows, pooled_a, pooled_b = [], [], []
    for n, plans, e in [(10, ["57a"], "n10_severe_with_poisoning"), (25, ["57a"], "n25_severe_with_poisoning"),
                        (50, ["57b1", "57b2"], "n50_severe_with_poisoning")]:
        a, b = cells(plans, e, "global_prox_decay"), cells(plans, e, "global_prox_nodecay")
        rows.append([f"{n} nodes", ms(a)[0], ms(b)[0], paired(a, b)])
        for s in sorted(set(a) & set(b)):
            pooled_a.append(a[s]); pooled_b.append(b[s])
    d = [x - y for x, y in zip(pooled_a, pooled_b)]
    t = stats.ttest_rel(pooled_a, pooled_b)
    sign = stats.binomtest(sum(x > 0 for x in d), sum(x != 0 for x in d)).pvalue
    rows.append([f"pooled ({len(d)} pairs)", "", "", f"{sum(d)/len(d):+.2f} (t={t.statistic:.2f}, p={t.pvalue:.3f}; "
                 f"{sum(x > 0 for x in d)} +, {sum(x < 0 for x in d)} −, smallest |diff| {min(abs(x) for x in d):.2f}; "
                 f"sign test p={sign:.2f})"])
    table("Age factor in aggregation vs. network size", "plans 57a (10, 25 nodes), 57b1+57b2 (50 nodes), severe + 20% poisoning",
          ["size", "λ = 1", "λ = 0", "λ1 − λ0"], rows)


# ------------------------------------------------------------------ Table scale (plan 63a vs 60c)
def t_scale():
    rows = []
    for arm in ["global_prox", "lightyear", "balance", "rfa", "robust"]:
        c10 = cells(["60c"], "3_non_iid_severe_with_poisoning", arm, fill_from=["60cr"])
        c25 = cells(["63a"], "n25_severe_with_poisoning", arm)
        c50 = cells(["63b1", "63b2"], "n50_severe_with_poisoning", arm)
        rows.append([arm, ms(c10)[0], ms(c25)[0], ms(c50)[0], "—" if arm == "global_prox" else
                     paired(cells(["63a"], "n25_severe_with_poisoning", "global_prox"), c25),
                     "—" if arm == "global_prox" else
                     paired(cells(["63b1", "63b2"], "n50_severe_with_poisoning", "global_prox"), c50)])
    table("Table scale: 10, 25 and 50 nodes", "10 nodes: plan 60c (+60cr); 25: plan 63a; 50: plans 63b1 + 63b2; all molab",
          ["aggregator", "10 nodes", "25 nodes", "50 nodes", "gp − row (25)", "gp − row (50)"], rows)


def t_alie():
    arms = ["global_prox", "lightyear", "balance", "rfa", "robust"]
    ref = {a: cells(["62"], "sigma_10p0", a) for a in arms}
    rows = []
    for e in ["alie_baruch", "alie_z1", "ipm_eps0p1", "ipm_eps1"]:
        c = {a: cells(["64"], e, a) for a in arms}
        drop = {a: sum(c[a][k] - ref[a][k] for k in c[a]) / len(c[a]) for a in arms}
        rows.append([e] + [f"{ms(c[a])[0]} ({drop[a]:+.1f})" for a in arms] + [paired(c["global_prox"], c["lightyear"])])
    table("Table alie: structured in-band attacks", "plan 64 (molab); in parentheses: mean change against the σ=10 cell "
          "of plan 62 (same platform, oracle setting and seeds)", ["attack"] + arms + ["global_prox − lightyear"], rows)


# ------------------------------------------------------------------ Full ledger (plan 65; plan 66 gates arm)
def t_full_ledger():
    nodes = pd.read_csv(HERE / "nodes_final.csv")
    E, H = "3_non_iid_severe_with_poisoning_t2rc", "2_non_iid_severe_honest_t2rc"
    rows = []
    for arm in ["global_prox", "global_prox_norep", "prox", "balance", "lightyear", "robust", "rfa"]:
        sel = ok[(ok.plan == "65") & (ok.experiment == E) & (ok.arm == arm)]
        nd = nodes[nodes.run_id.isin(sel.run_id)]
        rows.append([arm, ms(cells(["65"], E, arm))[0],
                     "—" if arm == "global_prox" else paired(cells(["65"], E, "global_prox"), cells(["65"], E, arm)),
                     f"{int(sel.slashed_malicious.sum())} / {int((nd.node_type != 'honest').sum())}",
                     int(sel.slashed_honest.sum()),
                     f"{nd[nd.node_type != 'honest'].reputation_score.mean():.2f} / {nd[nd.node_type == 'honest'].reputation_score.mean():.2f}",
                     ms(cells(["65"], H, arm))[0] if arm in ("global_prox", "global_prox_norep", "robust") else ""])
    hs = ok[(ok.plan == "65") & (ok.experiment == H)]
    rows.append(["honest control: honest nodes excluded", "", "", "", int(hs.slashed_honest.sum()), "", f"{len(hs)} runs"])
    g = ok[(ok.plan == "66") & (ok.experiment == "alie_z1_t2rc_gates")]
    rows.append(["plan 66, ALIE z=1, gates: attackers excluded", "", "", f"{int(g.slashed_malicious.sum())}", int(g.slashed_honest.sum()), "", f"{len(g)} runs"])
    table("Full ledger: exclusion layer on (Sec. inside the merge, paragraph 'With the exclusion layer on')",
          "plan 65 (molab; T2RC distance gates, max_strikes 1; compare only within the plan); last row plan 66",
          ["aggregator", "severe + poisoning", "gp − row", "attackers excluded", "honest excluded",
           "final rep. attacker / honest", "honest control"], rows)


# ------------------------------------------------------------------ mu sweep, lambda_select (plan 67a), ClippedGossip under ALIE (68)
def t_mu_lsel():
    ref = cells(["67a"], "mu_sweep", "mu_0p5")
    rows = []
    for mu in ["0p001", "0p01", "0p1", "0p25", "0p5", "1p0", "2p0", "4p0"]:
        c = cells(["67a"], "mu_sweep", f"mu_{mu}")
        rows.append([mu.replace("p", "."), ms(c)[0], "—" if mu == "0p5" else paired(c, ref)])
    same = ref == cells(["60c"], "3_non_iid_severe_with_poisoning", "global_prox")
    table("mu sweep: Farabi-Prox, severe + 20% Gaussian poisoning", f"plan 67a (molab); mu=0.5 identical to 60c: {same}",
          ["mu", "accuracy", "row − (mu=0.5)"], rows)
    ref = cells(["67a"], "mu_alie", "mu_0p5")
    rows = [[mu.replace("p", "."), ms(cells(["67a"], "mu_alie", f"mu_{mu}"))[0],
             "—" if mu == "0p5" else paired(cells(["67a"], "mu_alie", f"mu_{mu}"), ref)] for mu in ["0p5", "2p0", "4p0"]]
    table("mu under ALIE z=1", f"plan 67a; mu=0.5 identical to plan 64 alie_z1 global_prox: {ref == cells(['64'], 'alie_z1', 'global_prox')}",
          ["mu", "accuracy", "row − (mu=0.5)"], rows)
    rows = []
    for label, e1, a1, e0 in [("severe + poisoning", "mu_sweep", "mu_0p5", "lsel_severe_with_poisoning"),
                              ("severe, honest", "lsel_severe_honest", "lsel_1", "lsel_severe_honest")]:
        c1, c0 = cells(["67a"], e1, a1), cells(["67a"], e0, "lsel_0")
        rows.append([label, ms(c1)[0], ms(c0)[0], paired(c1, c0)])
    table("Age factor in tip selection (lambda_select 1 vs 0, lambda_aggregate 1)", "plan 67a (molab)",
          ["scenario", "λ_sel = 1", "λ_sel = 0", "difference"], rows)
    c, g = cells(["68"], "alie_z1", "clippedgossip"), cells(["68"], "alie_z1", "global_prox")
    table("ClippedGossip under ALIE z=1", "plan 68 (oracle true accuracy on, as plan 64)",
          ["arm", "accuracy", "clippedgossip − global_prox"], [["clippedgossip", ms(c)[0], paired(c, g)], ["global_prox", ms(g)[0], "—"]])


# ------------------------------------------------------------------ gossip_k vs num_tips (plan 67b)
def t_view():
    corners = [("k6_t3", "gossip_k 6, ≤3 parents (= 60c)"), ("k9_t3", "gossip_k 9, ≤3 parents"),
               ("k6_t9", "gossip_k 6, ≤9 parents"), ("k9_t9", "gossip_k 9, ≤9 parents (= 60d)")]
    rows = []
    for e, label in corners:
        g, f, d = (cells(["67b"], e, a) for a in ("global_prox", "rfa", "robust"))
        sel = ok[(ok.plan == "67b") & (ok.experiment == e) & (ok.arm == "global_prox")]
        pm = merge[merge.run_id.isin(sel.run_id)].frac_merges_poisoned_majority.mean() if merge is not None else float("nan")
        rows.append([label, ms(g)[0], ms(f)[0], ms(d)[0], paired(g, f), paired(g, d), f"{pm:.3f}"])
    rows.append(["change vs (6, 3): global_prox", "", "", "",
                 "; ".join(f"{e}: {paired(cells(['67b'], e, 'global_prox'), cells(['67b'], 'k6_t3', 'global_prox'))}"
                           for e, _ in corners[1:]), "", ""])
    rows.append(["change vs (6, 3): rfa", "", "", "",
                 "; ".join(f"{e}: {paired(cells(['67b'], e, 'rfa'), cells(['67b'], 'k6_t3', 'rfa'))}"
                           for e, _ in corners[1:]), "", ""])
    same = all(cells(["67b"], "k6_t3", a) == cells(["60c"], "3_non_iid_severe_with_poisoning", a) and
               cells(["67b"], "k9_t9", a) == cells(["60d"], "3w_non_iid_severe_with_poisoning_wide", a)
               for a in ("global_prox", "rfa", "robust"))
    table("View width: gossip_k and num_tips separated, severe + 20% poisoning",
          f"plan 67b (molab); corners (6,3) and (9,9) identical to 60c / 60d in every run: {same}",
          ["corner", "global_prox", "rfa", "robust (distance gates)", "gp − rfa", "gp − robust",
           "share of honest merges with poisoned majority (global_prox runs)"], rows)


# ------------------------------------------------------------------ second review: Holm, worst node, view diagnostics
def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def _p(a, b):
    s = sorted(set(a) & set(b))
    return stats.ttest_rel([a[k] for k in s], [b[k] for k in s]).pvalue


def t_holm():
    """Two families: (1) the headline scenario, Farabi-Prox (self) against each other aggregator;
    (2) every paired comparison of Table main (4 scenarios x 9)."""
    others = ARMS[1:] + ["clippedgossip"]
    def pair(scen, arm):
        _, p, e, fill = scen
        if arm == "clippedgossip":
            return cells(["68"], e, "global_prox"), cells(["68"], e, "clippedgossip")
        return cells([p], e, "global_prox", fill), cells([p], e, arm, fill)
    fam1 = [(a, *pair(SCEN[2], a)) for a in others]
    raw1 = [_p(g, o) for _, g, o in fam1]
    adj1 = holm(raw1)
    rows = [[f"global_prox − {a}", paired(g, o), f"{r:.4f}", f"{q:.4f}", "yes" if q < 0.05 else "no"]
            for (a, g, o), r, q in zip(fam1, raw1, adj1)]
    table("Holm correction, family 1: headline scenario (Severe + poisoning), 9 comparisons",
          "as Table main; two-sided paired t-tests, Holm step-down over the family",
          ["comparison", "paired difference", "raw p", "Holm p", "significant at 0.05 after Holm"], rows)
    fam2 = [(sc[0], a, *pair(sc, a)) for sc in SCEN for a in others]
    raw2 = [_p(g, o) for *_, g, o in fam2]
    adj2 = holm(raw2)
    rows = [[f"{sc}: global_prox − {a}", paired(g, o), f"{r:.4f}", f"{q:.4f}", "yes" if q < 0.05 else "no"]
            for (sc, a, g, o), r, q in zip(fam2, raw2, adj2)]
    table("Holm correction, family 2: all 36 paired comparisons of Table main",
          "as Table main; two-sided paired t-tests, Holm step-down over all 36",
          ["comparison", "paired difference", "raw p", "Holm p", "significant at 0.05 after Holm"], rows)


def cells_min(plans, experiment, arm, fill_from=()):
    """{seed: accuracy of the worst honest node at t = 600}."""
    sel = ok[ok.plan.isin(plans) & (ok.experiment == experiment) & (ok.arm == arm)]
    out = dict(zip(sel.seed, sel.final_honest_acc_min))
    for p in fill_from:
        extra = ok[(ok.plan == p) & (ok.experiment == experiment) & (ok.arm == arm)]
        for sd, v in zip(extra.seed, extra.final_honest_acc_min):
            out.setdefault(sd, v)
    return out


def t_worst():
    rows = []
    for arm in ARMS:
        rows.append([arm] + [ms(cells_min([p], e, arm, fill))[0] for _, p, e, fill in SCEN])
    rows.append(["clippedgossip (plan 68)"] + [ms(cells_min(["68"], e, "clippedgossip"))[0] for _, p, e, fill in SCEN])
    table("Worst honest node: accuracy of the least accurate honest node at t = 600, MNIST",
          "runs of Table main; mean ± sample SD over seeds of the per-run minimum over honest nodes",
          ["aggregator"] + [s for s, *_ in SCEN], rows)


def t_view_diag():
    corners = ["k6_t3", "k9_t3", "k6_t9", "k9_t9"]
    rows = []
    for e in corners:
        sel = ok[(ok.plan == "67b") & (ok.experiment == e)]
        m = merge[merge.run_id.isin(sel.run_id)].merge(sel[["run_id", "arm"]], on="run_id")
        g = m[m.arm == "global_prox"]
        rows.append([e, f"{g.merges_honest.mean():.0f}",
                     " / ".join(f"{m[m.arm == a].merges_honest.mean():.0f}" for a in ("rfa", "robust")),
                     f"{g.neff_mean_k0.mean():.2f}", f"{g.neff_mean_k1.mean():.2f}", f"{g.neff_mean_k2.mean():.2f}",
                     f"{g.dist_honest_p50.mean():.2f}", ms(cells_min(["67b"], e, "global_prox"))[0]])
    table("View width, merge diagnostics (Farabi-Prox runs unless stated)",
          "plan 67b, merge_stats.csv; merges = honest nodes' merges per run; n_eff by number k of poisoned parents",
          ["corner", "merges (global_prox)", "merges (rfa / robust)", "n_eff k=0", "n_eff k=1", "n_eff k=2",
           "median honest-parent distance", "worst honest node (global_prox)"], rows)


# ------------------------------------------------------------------ Table ironforge (plan 69, 2x2)
def t_ironforge():
    arms = [("ironforge_iftrain", "IronForge, its training"), ("ironforge_ourtrain", "IronForge, our training"),
            ("global_prox_iftrain", "Farabi-Prox, IronForge's training"), ("global_prox", "Farabi-Prox, our training")]
    scen = [("Moderate", "1_non_iid_moderate_honest"), ("Severe", "2_non_iid_severe_honest"),
            ("Severe + poisoning", "3_non_iid_severe_with_poisoning"),
            ("Severe + poisoning, wide", "3w_non_iid_severe_with_poisoning_wide")]
    rows = []
    for name, e in scen:
        c = {a: cells(["69"], e, a) for a, _ in arms}
        rows.append([name] + [ms(c[a])[0] for a, _ in arms] +
                    [paired(c["global_prox_iftrain"], c["ironforge_iftrain"]), paired(c["global_prox"], c["ironforge_ourtrain"]),
                     paired(c["global_prox"], c["global_prox_iftrain"]), paired(c["ironforge_ourtrain"], c["ironforge_iftrain"])])
    ref = {e: p for p, e in zip(["60a", "60b", "60c", "60d"], [e for _, e in scen])}
    same = all(cells(["69"], e, "global_prox") == cells([ref[e]], e, "global_prox") for _, e in scen)
    table("Table ironforge: protocol x training, MNIST",
          f"plan 69 (molab), one experiment; Farabi-Prox/our training identical to 60a-d in every run: {same}",
          ["scenario"] + [l for _, l in arms] + ["GP − IF (IF training)", "GP − IF (our training)",
                                                  "GP: ours − IF's", "IF: ours − its own"], rows)
    rows = []
    for name, e in scen:
        for a, lab in arms[:2]:
            sel = ok[(ok.plan == "69") & (ok.experiment == e) & (ok.arm == a)]
            att = f"{int(sel.slashed_malicious.sum())} of {int((sel.n_nodes * sel.frac_poisoning).sum())}" \
                if sel.frac_poisoning.fillna(0).sum() else "—"
            rows.append([f"{name}, {lab}", att, f"{int(sel.slashed_honest.sum())} of {int(sel.n_honest.sum())}"])
    table("IronForge exclusions", "plan 69", ["scenario, arm", "attackers excluded", "honest nodes excluded"], rows)


# ------------------------------------------------------------------ Table cifar (plans 58a, 58b, 58d, 55)
def t_cifar():
    e = "S1_cifar_severe_with_poisoning"
    ref = cells(["58a", "58b"], e, "global_prox")
    rows = []
    for arm in ["global_prox", "balance", "lightyear", "robust", "multi_krum", "rfa"]:
        c = cells(["58a", "58b"], e, arm)
        rows.append([arm, ms(c)[0], "—" if arm == "global_prox" else paired(ref, c)])
    ir = cells(["58d"], e, "ironforge")
    sel = ok[(ok.plan == "58d")]
    rows.append(["IronForge (protocol, plan 58d)", ms(ir)[0],
                 f"slashed {int(sel.slashed_malicious.sum())} attackers, {int(sel.slashed_honest.sum())} honest"])
    table("Table cifar: CIFAR-10, severe + 20% poisoning", "plans 58a (seeds 42/7/44) + 58b (13/99), Kaggle; IronForge 58d",
          ["aggregator", "accuracy", "global_prox − row"], rows)
    rows = []
    gp5, rb5, cz5 = (cells(["58c"], "S2_cifar_moderate_honest", a) for a in ("global_prox", "robust", "centralized_fedavg"))
    rows.append(["α=0.5 honest: global_prox", ms(gp5)[0], "—"])
    rows.append(["α=0.5 honest: robust", ms(rb5)[0], paired(gp5, rb5)])
    rows.append(["α=0.5 honest: centralized FedAvg", ms(cz5)[0], paired(gp5, cz5)])
    rows.append(["α=0.1 honest: centralized FedAvg (reference)", ms(cells(["58c"], "S3_cifar_severe_honest", "centralized_fedavg"))[0], "—"])
    table("CIFAR-10 references (plan 58c)", "plan 58c (molab), 5 seeds; the α=0.1 decentralized arms are in 58a/58b "
          "(Kaggle), so the α=0.1 centralized value is a reference line only", ["arm", "accuracy", "global_prox − row"], rows)
    rows = []
    for e, g in ok[ok.plan == "55"].groupby("experiment"):
        rows.append([e, "; ".join(f"{a} {v:.1f}" for a, v in zip(g.arm, g.final_honest_acc))])
    table("CIFAR-10 calibration (one seed)", "plan 55 (Kaggle), seed 42", ["scenario", "final accuracy by arm"], rows)


# ------------------------------------------------------------------ Table inband (plan 62)
def t_inband():
    arms = ["global_prox", "rfa", "robust", "lightyear", "balance"]
    rows = []
    for e, label in [("sigma_10p0", "σ=10"), ("sigma_0p1", "σ=0.1"), ("sigma_0p01", "σ=0.01"), ("sigma_0p003", "σ=0.003")]:
        c = {a: cells(["62"], e, a) for a in arms}
        rows.append([label] + [ms(c[a])[0] for a in arms] + [paired(c["global_prox"], c["rfa"])])
    table("Table inband: Gaussian poisoning with decreasing noise", "plan 62 (molab), severe + 20% poisoning, "
          "oracle_true_acc on (own σ=10 reference)", ["σ"] + arms + ["global_prox − rfa"], rows)
    if merge is not None:
        m = merge.merge(runs[["run_id", "plan", "experiment", "arm"]], on="run_id")
        g = m[(m.plan == "62") & (m.arm == "global_prox")].groupby("experiment")
        rows = [[e, f"{x.dist_poisoned_p50.mean():.1f}", f"{x.dist_honest_p50.mean():.2f}", f"{x.weight_poisoned_mean.mean():.3f}"]
                for e, x in g]
        table("Poisoned-parent distance and weight in Farabi-Prox merges", "plan 62 merge diagnostics, mean of per-run values",
              ["experiment", "poisoned dist p50", "honest dist p50", "poisoned weight"], rows)


# ------------------------------------------------------------------ log-derived numbers
def t_logs():
    if merge is None:
        MD.append("\n(merge_stats.csv missing: run build_dataset.py without --no-logs)\n")
        return
    m = merge.merge(runs[["run_id", "plan", "experiment", "arm", "complete"]], on="run_id")
    m = m[m.complete == True]  # noqa: E712
    rows = []
    for plan, label in [("60c", "≤3 parents (60c)"), ("60d", "≤9 parents (60d)")]:
        g = m[m.plan == plan].groupby("arm").agg(agg_ms=("agg_ms_mean", "mean"), guard=("guard_fired", "sum"),
                                                  merges=("merges", "sum"))
        for arm, r in g.iterrows():
            rows.append([f"{arm} {label}", f"{r.agg_ms:.1f}", f"{int(r.guard)} / {int(r.merges)}"])
    table("Merge cost and trainer-guard firings", "simulation.log of plans 60c and 60d; cost = mean of per-run mean "
          "wall time of aggregate() (Supplementary Table S1 pools all merges, which differs slightly)",
          ["arm", "ms per merge", "guard fired / merges"], rows)
    rows = []
    for plan, e, arm in [("60c", "3_non_iid_severe_with_poisoning", "global_prox"),
                         ("57a", "n25_severe_with_poisoning", "global_prox_decay"),
                         ("57b1", "n50_severe_with_poisoning", "global_prox_decay"),
                         ("57b2", "n50_severe_with_poisoning", "global_prox_decay")]:
        g = m[(m.plan == plan) & (m.experiment == e) & (m.arm == arm)]
        rows.append([f"{plan} {e} {arm}", f"{g.frac_merges_poisoned_majority.mean():.3f}",
                     f"{g.frac_merges_all_poisoned.mean():.3f}", f"{g.agg_ms_mean.mean():.1f}"])
    table("Parent composition of honest merges", "simulation.log ([PROCESSOR] parent list + node types)",
          ["run group", "share with poisoned majority", "share all poisoned", "ms per merge"], rows)


# ------------------------------------------------------------------ baseline sensitivity (plan 71)
def t_sensitivity():
    E, H = "severe_with_poisoning", "severe_honest"
    gp = cells(["71"], E, "global_prox")
    gpb = cells(["60b"], "2_non_iid_severe_honest", "global_prox")
    arms = [("balance_g0p1", "BALANCE γ=0.1"), ("balance", "BALANCE γ=0.3, α=0.5 (paper)"),
            ("balance_g1p0", "BALANCE γ=1.0"), ("balance_a0p1", "BALANCE α=0.1"),
            ("lightyear_t0p5", "LIGHTYEAR τ=0.5"), ("lightyear_t0p6", "LIGHTYEAR τ=0.6"),
            ("lightyear", "LIGHTYEAR τ=0.75 (paper)"), ("lightyear_t0p9", "LIGHTYEAR τ=0.9")]
    rows = [["Farabi-Prox", ms(gp)[0], "—", ms(gpb)[0] + " (60b)", "—"]]
    for a, label in arms:
        c, h = cells(["71"], E, a), cells(["71"], H, a)
        rows.append([label, ms(c)[0], paired(gp, c), ms(h)[0] if h else "—", paired(gpb, h) if h else "—"])
    same = all(cells(["71"], E, a) == cells(["60c"], "3_non_iid_severe_with_poisoning", a, ["60cr"])
               for a in ("global_prox", "balance", "lightyear")) and \
        cells(["71"], H, "balance") == cells(["60b"], "2_non_iid_severe_honest", "balance")
    table("Baseline sensitivity: BALANCE and LIGHTYEAR hyperparameters",
          f"plan 71 (molab), 5 seeds; default arms identical to 60c/60cr and 60b in every run: {same} "
          "(so the severe-honest column is compared with Farabi-Prox of 60b, same platform)",
          ["arm", "severe + poisoning", "Farabi-Prox − row", "severe, honest", "Farabi-Prox (60b) − row"], rows)


# ------------------------------------------------------------------ headline cell with ten seeds (60c + 60cr + 73)
def t_headline10():
    e = SCEN[2][2]
    ten = {a: cells(["60c", "73"], e, a, ["60cr"]) for a in ("global_prox", "prox", "lightyear", "balance")}
    g = ten["global_prox"]
    rows = [[a, ms(c)[0], ms(c)[1], "—" if a == "global_prox" else paired(g, c)] for a, c in ten.items()]
    table("Headline cell with ten seeds (Severe + poisoning)",
          "plans 60c (seeds 42/7/44/13/99, LIGHTYEAR seed 99 from 60cr) + 73 (seeds 11/23/57/77/101), molab",
          ["arm", "accuracy", "n", "global_prox − row"], rows)
    others = ARMS[1:] + ["clippedgossip"]
    def pair(scen, arm):
        _, p, ex, fill = scen
        if arm == "clippedgossip":
            return cells(["68"], ex, "global_prox"), cells(["68"], ex, "clippedgossip")
        if scen is SCEN[2] and arm in ten:
            return g, ten[arm]
        return cells([p], ex, "global_prox", fill), cells([p], ex, arm, fill)
    for name, fam in [("family 1 (headline, 9)", [(SCEN[2], a) for a in others]),
                      ("family 2 (Table main, 36)", [(sc, a) for sc in SCEN for a in others])]:
        prs = [pair(sc, a) for sc, a in fam]
        raw = [_p(x, y) for x, y in prs]
        adj = holm(raw)
        rows = [[f"{sc[0]}: global_prox − {a}", paired(x, y), f"{r:.4f}", f"{q:.4f}", "yes" if q < 0.05 else "no"]
                for (sc, a), (x, y), r, q in zip(fam, prs, raw, adj) if sc is SCEN[2] and a in ten]
        n_sig = sum(q < 0.05 for q in adj)
        rows.append([f"significant after Holm over the whole {name}", f"{n_sig} of {len(fam)}", "", "", ""])
        table(f"Holm with ten seeds in the headline cell, {name}",
              "as the Holm tables above, but prox / lightyear / balance in Severe + poisoning use the ten seeds",
              ["comparison", "paired difference", "raw p", "Holm p", "significant at 0.05 after Holm"], rows)


# ------------------------------------------------------------------ age factor 2x2 (plan 72)
def t_age2x2():
    labels = {"l1_s1": "λ_agg 1, λ_sel 1 (default)", "l0_s1": "λ_agg 0, λ_sel 1",
              "l1_s0": "λ_agg 1, λ_sel 0", "l0_s0": "λ_agg 0, λ_sel 0"}
    rows = []
    for e, name in [("severe_with_poisoning", "severe + poisoning"), ("severe_honest", "severe, honest")]:
        A = {a: cells(["72"], e, a) for a in labels}
        for a, lab in labels.items():
            rows.append([f"{name}: {lab}", ms(A[a])[0], ""])
        k = sorted(A["l1_s1"])
        inter = [(A["l1_s1"][s] - A["l0_s1"][s]) - (A["l1_s0"][s] - A["l0_s0"][s]) for s in k]
        rows += [[f"{name}: λ_agg effect at λ_sel 1", "", paired(A["l1_s1"], A["l0_s1"])],
                 [f"{name}: λ_agg effect at λ_sel 0", "", paired(A["l1_s0"], A["l0_s0"])],
                 [f"{name}: λ_sel effect at λ_agg 1", "", paired(A["l1_s1"], A["l1_s0"])],
                 [f"{name}: λ_sel effect at λ_agg 0", "", paired(A["l0_s1"], A["l0_s0"])],
                 [f"{name}: interaction", "", f"{sum(inter)/len(inter):+.2f} (p={stats.ttest_1samp(inter, 0).pvalue:.3f})"]]
    same = cells(["72"], "severe_with_poisoning", "l1_s1") == cells(["60c"], SCEN[2][2], "global_prox") and \
        cells(["72"], "severe_honest", "l1_s1") == cells(["60b"], SCEN[1][2], "global_prox") and \
        cells(["72"], "severe_with_poisoning", "l1_s0") == cells(["67a"], "lsel_severe_with_poisoning", "lsel_0")
    table("Age factor 2x2: aggregation (λ_agg) x tip selection (λ_sel), Farabi-Prox",
          f"plan 72 (molab), one experiment; controls identical to 60c, 60b and 67a lsel_0: {same}",
          ["arm / contrast", "accuracy", "paired difference"], rows)
    a, b = [], []
    for plans, e, x, y in [(["57a"], "n10_severe_with_poisoning", "global_prox_decay", "global_prox_nodecay"),
                           (["57a"], "n25_severe_with_poisoning", "global_prox_decay", "global_prox_nodecay"),
                           (["57b1", "57b2"], "n50_severe_with_poisoning", "global_prox_decay", "global_prox_nodecay"),
                           (["72"], "severe_with_poisoning", "l1_s1", "l0_s1")]:
        ca, cb = cells(plans, e, x), cells(plans, e, y)
        for s in sorted(set(ca) & set(cb)):
            a.append(ca[s]); b.append(cb[s])
    d = [p - q for p, q in zip(a, b)]
    t = stats.ttest_rel(a, b)
    sign = stats.binomtest(sum(v > 0 for v in d), sum(v != 0 for v in d)).pvalue
    table("Age factor in aggregation, pooled pairs including plan 72",
          "within-plan seed pairs of 57a (10, 25 nodes, Kaggle), 57b1+57b2 (50 nodes), 72 (10 nodes, molab)",
          ["pairs", "mean λ1 − λ0", "test"],
          [[len(d), f"{sum(d)/len(d):+.2f}", f"t={t.statistic:.2f}, p={t.pvalue:.3f}; {sum(v > 0 for v in d)} +, "
            f"{sum(v < 0 for v in d)} −; sign test p={sign:.2f}"]])


# ------------------------------------------------------------------ kernel ablation: exp kernel vs hard filter (plan 70)
def t_kernel():
    scen = [("severe_with_poisoning", "severe + 20% poisoning (σ=10)"), ("severe_honest", "severe, honest"),
            ("sigma_0p01", "σ=0.01 (poisoned ≈ 6.6 units)"), ("sigma_0p003", "σ=0.003 (poisoned ≈ 2.2 units)")]
    rows = []
    for e, name in scen:
        g = cells(["70"], e, "global_prox")
        row = [name, ms(g)[0]]
        for a in ("hard_r2", "hard_r4", "hard_r50"):
            h = cells(["70"], e, a)
            row += [ms(h)[0], paired(h, g)]
        rows.append(row)
    same = cells(["70"], "severe_with_poisoning", "global_prox") == cells(["60c"], SCEN[2][2], "global_prox") and \
        cells(["70"], "severe_honest", "global_prox") == cells(["60b"], SCEN[1][2], "global_prox") and \
        all(cells(["70"], e, "global_prox") == cells(["62"], e, "global_prox") for e in ("sigma_0p01", "sigma_0p003"))
    table("Kernel ablation: exponential kernel vs hard filter around the own model",
          f"plan 70 (molab); hard = reputation x 1[d <= r] x age, zero-weight fallback uniform; σ cells oracle_true_acc on (as 62); "
          f"exp arms identical to 60c, 60b and 62: {same}",
          ["scenario", "exp kernel (μ=0.5)", "hard r=2", "r2 − exp", "hard r=4", "r4 − exp", "hard r=50", "r50 − exp"], rows)
    if merge is not None:
        m = merge.merge(runs[["run_id", "plan", "experiment", "arm"]], on="run_id")
        g = m[(m.plan == "70")].groupby(["experiment", "arm"])[["guard_fired_honest", "merges"]].sum()
        rows = [[f"{e} {a}", f"{int(r.guard_fired_honest)} / {int(r.merges)}"] for (e, a), r in g.iterrows() if r.guard_fired_honest]
        table("Kernel ablation: trainer-guard firings in honest merges", "plan 70 simulation.log (only arms with firings)",
              ["arm", "guard fired / merges"], rows)


# ------------------------------------------------------------------ entry reputation x reward scale (plans 76a, 76b)
def t_entry_rep():
    scen = [("76a", "severe_with_poisoning", "severe + 20% poisoning (σ=10)"), ("76a", "severe_honest", "severe, honest"),
            ("76b", "alie_z1", "ALIE z=1")]
    arms = ["const_gp", "const_gp_norep", "const_prox", "stake_gp", "stake_gp_norep", "stake_prox",
            "sybil_gp", "sybil_gp_norep", "sybil_prox", "const_gp_r15", "stake_prox_r15", "sybil_gp_r15", "sybil_prox_r15"]
    rows = []
    for p, e, name in scen:
        for a in arms:
            v = cells([p], e, a)
            if v:
                rows.append([f"{name}: {a}", ms(v)[0]])
    same = cells(["76a"], "severe_with_poisoning", "const_gp") == cells(["60c"], SCEN[2][2], "global_prox") and \
        cells(["76a"], "severe_honest", "const_gp") == cells(["60b"], SCEN[1][2], "global_prox") and \
        cells(["76b"], "alie_z1", "const_gp") == cells(["64"], "alie_z1", "global_prox")
    table("Entry reputation and reward scale: accuracy per arm",
          f"plans 76a + 76b (molab, run in parallel); const_gp identical to 60c, 60b and 64 alie_z1: {same}; "
          "stake = random log-uniform stake, sybil = attackers at the minimum stake, _norep = reputation_weight false, "
          "_r15 = reward_scale 0.15 (default 0.01)", ["scenario: arm", "accuracy"], rows)
    comps = [("reputation factor (stake)", "stake_gp", "stake_gp_norep"), ("reputation factor (sybil)", "sybil_gp", "sybil_gp_norep"),
             ("reputation anchor, sybil vs constant", "sybil_prox", "const_prox"),
             ("reputation anchor (sybil) vs self anchor (constant)", "sybil_prox", "const_gp"),
             ("reputation anchor, stake vs constant", "stake_prox", "const_prox"),
             ("reward 0.15 vs 0.01, sybil anchor", "sybil_prox_r15", "sybil_prox")]
    rows = []
    for p, e, name in [s for s in scen if s[1] != "severe_honest"]:
        for lab, a, b in comps:
            rows.append([f"{name}: {lab}", paired(cells([p], e, a), cells([p], e, b))])
    table("Entry reputation: paired contrasts", "plans 76a, 76b", ["contrast", "row − reference"], rows)


# ------------------------------------------------------------------ ten seeds for the top arms in all scenarios (73, 79) and WFAgg (78)
TEN = {"60a": "79", "60b": "79", "60c": "73", "60d": "79"}


def ten_cells(scen, arm):
    """Ten-seed cells where plan 73/79 added seeds for this arm, else the five-seed cells of Table main."""
    name, p, e, fill = scen
    extra = TEN[p]
    c = cells([p, extra], e, arm, fill)
    return c if len(c) >= 10 else cells([p], e, arm, fill)


def t_ten_seeds():
    arms = ["global_prox", "prox", "lightyear", "balance", "rfa"]
    rows = []
    for scen in SCEN:
        g = ten_cells(scen, "global_prox")
        for a in arms:
            c = ten_cells(scen, a)
            if a == "rfa" and len(c) < 10:
                continue
            rows.append([f"{scen[0]}: {a}", ms(c)[0], ms(c)[1], "—" if a == "global_prox" else paired(g, c)])
    table("Ten seeds for the aggregators that keep learning (Table main, all four scenarios)",
          "plans 60a-d (+60abr/60cr fill) and 73 (60c) / 79 (60a, 60b, 60d), seeds 42/7/44/13/99 + 11/23/57/77/101, molab",
          ["scenario: arm", "accuracy", "n", "global_prox − row"], rows)
    others = ARMS[1:] + ["clippedgossip"]
    fam, prs = [], []
    for scen in SCEN:
        for a in others:
            if a == "clippedgossip":
                x, y = cells(["68"], scen[2], "global_prox"), cells(["68"], scen[2], "clippedgossip")
            else:
                x, y = ten_cells(scen, "global_prox"), ten_cells(scen, a)
                if len(y) < 10 or len(x) < 10:
                    x, y = cells([scen[1]], scen[2], "global_prox", scen[3]), cells([scen[1]], scen[2], a, scen[3])
            fam.append((scen[0], a)); prs.append((x, y))
    raw = [_p(x, y) for x, y in prs]
    adj = holm(raw)
    rows = [[f"{sc}: global_prox − {a}", paired(x, y), f"{r:.4f}", f"{q:.4f}", "yes" if q < 0.05 else "no"]
            for (sc, a), (x, y), r, q in zip(fam, prs, raw, adj)
            if len(ten_cells(next(s for s in SCEN if s[0] == sc), a)) >= 10 or a == "global_prox"]
    rows.append(["significant after Holm over all 36 (ten seeds where available)", f"{sum(q < 0.05 for q in adj)} of 36", "", "", ""])
    table("Holm over all 36 comparisons of Table main, ten seeds where available", "as above",
          ["comparison", "paired difference", "raw p", "Holm p", "significant"], rows)
    # WFAgg sensitivity (plan 78)
    rows = []
    for e, name in [("severe_with_poisoning", "severe + poisoning"), ("severe_honest", "severe, honest")]:
        g = cells(["78"], e, "global_prox") or cells(["60b"], SCEN[1][2], "global_prox")
        for a in ("wfagg", "wfagg_f1", "wfagg_f0", "wfagg_a0p5", "wfagg_minf1"):
            v = cells(["78"], e, a)
            if v:
                rows.append([f"{name}: {a}", ms(v)[0], paired(g, v)])
    table("WFAgg sensitivity", "plan 78 (molab); f is clamped to floor((K-2)/2) = 0 for three parents, so f has no effect",
          ["arm", "accuracy", "Farabi-Prox − row"], rows)


for f in (t_main, t_prefilter, t_reputation, t_inband, t_alie, t_staleness, t_scale, t_full_ledger, t_mu_lsel, t_view,
          t_ironforge, t_cifar, t_logs, t_holm, t_worst, t_view_diag, t_sensitivity, t_headline10, t_age2x2, t_kernel, t_entry_rep, t_ten_seeds):
    f()
head = ["# Q1 paper numbers, recomputed from the dataset", "",
        "Generated by `reproduce_paper.py` from `runs.csv` and `merge_stats.csv`. Accuracy = mean over honest",
        "nodes of the full-test-set accuracy at the end of the run (t = 600 s); ± = sample SD over seeds;",
        "paired cells = mean difference (seeds with a positive difference / seeds, two-sided paired t-test).",
        "Only complete runs are used."]
(HERE / "paper_numbers.md").write_text("\n".join(head + MD) + "\n")
pd.DataFrame(CELLS).to_csv(HERE / "paper_numbers.csv", index=False)
print("wrote paper_numbers.md and paper_numbers.csv")
