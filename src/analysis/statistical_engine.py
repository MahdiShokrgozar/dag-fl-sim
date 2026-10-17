# src/analysis/statistical_engine.py

import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def _strip_mean_suffix(column: str) -> str:
    """Drops a trailing '_mean' only -- see plotter._strip_mean_suffix."""
    return column[: -len("_mean")] if column.endswith("_mean") else column

class StatEngine:
    """
    Statistical Engine for Hypothesis Testing and Confidence Intervals.
    
    This class performs rigorous statistical analysis on aggregated simulation data
    to determine if the observed differences between protocols (e.g., FARABI vs. FedAvg)
    are statistically significant.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): Significance level (usually 0.05 for a 95% confidence interval).
        """
        self.alpha = alpha
        self.results_history: List[Dict[str, Any]] = []

    def compute_significance(self, df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, 
                             time_point: Optional[float] = None) -> Dict[str, Any]:
        """
        Compares two distributions (A and B) for a specific metric to compute p-value.
        
        Args:
            df_a (pd.DataFrame): Aggregated data for Protocol A (e.g., FARABI).
            df_b (pd.DataFrame): Aggregated data for Protocol B (e.g., Baseline).
            metric (str): The column name to compare (e.g., 'global_mean_accuracy_mean').
            time_point (float, optional): The specific clock time to compare. 
                                          If None, compares the final simulation time.
                                          
        Returns:
            Dict[str, Any]: Statistical summary including p-value, t-statistic, and significance flag.
        """
        
        # 1. Align time points
        if time_point is None:
            # Default to the end of the simulation
            time_a = df_a['clock'].max()
            time_b = df_b['clock'].max()
            target_time = min(time_a, time_b)
        else:
            target_time = time_point

        # Find the closest rows to the target time
        row_a = self._get_closest_row(df_a, target_time)
        row_b = self._get_closest_row(df_b, target_time)
        
        if row_a is None or row_b is None:
             logger.warning(f"Could not find valid data at time ~{target_time} for metric {metric}")
             return {"significant": False, "p_value": 1.0, "error": "Missing data at time point"}

        # 2. Extract Mean and Standard Deviation (assuming they were pre-computed by aggregator)
        # `_strip_mean_suffix`, not str.replace: replace() drops EVERY '_mean',
        # so `global_mean_accuracy_mean` became `global_accuracy` and the std and
        # count lookups below silently missed. This project's headline metric is
        # exactly that one, so it was being dropped from every significance test
        # with "Insufficient variance data" while single-'_mean' metrics passed.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        mean_col = f"{metric}" if f"{metric}" in df_a.columns else metric
        base_col = _strip_mean_suffix(metric)
        std_col = f"{base_col}_std"
        
        if mean_col not in row_a or std_col not in row_a or pd.isna(row_a[std_col]):
             # If we don't have standard deviation (e.g., only 1 seed was run), we cannot perform a t-test
             logger.warning(f"Insufficient variance data for metric {metric}. Run multiple seeds.")
             return {"significant": False, "p_value": 1.0, "error": "Requires _std column from multiple seeds"}

        mean_a, std_a = row_a[mean_col], row_a[std_col]
        mean_b, std_b = row_b[mean_col], row_b[std_col]
        
        # Dynamically extract sample size (number of seeds), fallback to 5 for legacy data
        count_col = f"{base_col}_count"
        n_a = row_a[count_col] if count_col in row_a else 5
        n_b = row_b[count_col] if count_col in row_b else 5
        
        if n_a < 2 or n_b < 2:
            logger.warning(f"Sample size too small (N<2) for metric {metric} at t={target_time}. T-test requires variance.")
            return {"significant": False, "p_value": 1.0, "error": "Insufficient sample size (N<2)"}
        
        # 3. Perform Welch's t-test (assumes unequal variances) from descriptive statistics
        t_stat, p_value = stats.ttest_ind_from_stats(
            mean1=mean_a, std1=std_a, nobs1=n_a,
            mean2=mean_b, std2=std_b, nobs2=n_b,
            equal_var=False
        )
        
        is_significant = p_value < self.alpha
        effect = self._effect_size(mean_a, std_a, n_a, mean_b, std_b, n_b)
        ci_low, ci_high = self._difference_ci(mean_a, std_a, n_a, mean_b, std_b, n_b)

        result = {
            "metric": metric,
            "time_point": target_time,
            "mean_baseline": mean_a,
            "mean_competitor": mean_b,
            "difference": mean_a - mean_b,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "cohens_d": effect["cohens_d"],
            "hedges_g": effect["hedges_g"],
            "effect_magnitude": effect["magnitude"],
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "n_baseline": n_a,
            "n_competitor": n_b,
            "alpha": self.alpha
        }
        self.results_history.append(result)

        logger.info(
            f"Stats [{metric} @ t={target_time}]: Diff={mean_a - mean_b:.2f} "
            f"95%CI[{ci_low:.2f}, {ci_high:.2f}] g={effect['hedges_g']:.2f} "
            f"({effect['magnitude']}) p={p_value:.4f} -> Significant: {is_significant}"
        )
        return result

    @staticmethod
    def _effect_size(mean_a, std_a, n_a, mean_b, std_b, n_b) -> Dict[str, Any]:
        """
        Standardised difference between the two arms.

        A p-value answers "could this be chance?", not "is the gap big enough to
        matter" - and with only a handful of seeds those come apart badly: a
        12-point accuracy gap here sits at p=0.16 purely because n=3 leaves
        almost no power. Reporting an effect size alongside the p-value is what
        stops "not significant" from being read as "no difference".

        Hedges' g is the headline rather than Cohen's d because d is biased
        upward at small n, and n=3 per arm is exactly where that bias bites.

        CAVEAT worth knowing before quoting these in a paper: g divides by the
        pooled spread, so on metrics that barely vary across seeds - byte counts
        are the obvious case - it explodes. The 2026-09-12 crash check produced
        g=131 for total_ipfs_bytes, which is not a meaningful "standardised"
        effect, it is a near-zero denominator. For those metrics report the raw
        ratio (FARABI 515 MB vs IronForge 317 MB) and leave g out. g earns its
        place on noisy metrics like accuracy, where it separates "small gap" from
        "large gap we lack the seeds to confirm".
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        n_a, n_b = int(n_a), int(n_b)
        dof = n_a + n_b - 2
        if dof <= 0:
            return {"cohens_d": float("nan"), "hedges_g": float("nan"), "magnitude": "undefined"}

        pooled_var = ((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / dof
        pooled_sd = float(np.sqrt(pooled_var))
        if pooled_sd == 0:
            return {"cohens_d": float("nan"), "hedges_g": float("nan"), "magnitude": "undefined"}

        d = (mean_a - mean_b) / pooled_sd
        # Hedges' small-sample correction.
        g = d * (1.0 - 3.0 / (4.0 * dof - 1.0))

        a = abs(g)
        magnitude = "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"
        return {"cohens_d": float(d), "hedges_g": float(g), "magnitude": magnitude}

    def _difference_ci(self, mean_a, std_a, n_a, mean_b, std_b, n_b):
        """
        Welch confidence interval on the difference of means.

        Reported because the interval says what the p-value cannot: how large
        the gap plausibly is. An interval that comfortably spans zero is an
        honest "we cannot tell from this many seeds"; a narrow one far from zero
        is a claim worth making.
        """
        n_a, n_b = int(n_a), int(n_b)
        if n_a < 2 or n_b < 2:
            return float("nan"), float("nan")

        va, vb = std_a ** 2 / n_a, std_b ** 2 / n_b
        se = float(np.sqrt(va + vb))
        if se == 0:
            return mean_a - mean_b, mean_a - mean_b

        # Welch-Satterthwaite degrees of freedom.
        denom = (va ** 2) / (n_a - 1) + (vb ** 2) / (n_b - 1)
        dof = (va + vb) ** 2 / denom if denom > 0 else n_a + n_b - 2
        t_crit = float(stats.t.ppf(1.0 - self.alpha / 2.0, dof))
        diff = mean_a - mean_b
        return diff - t_crit * se, diff + t_crit * se

    def _get_closest_row(self, df: pd.DataFrame, target_time: float) -> Optional[pd.Series]:
        """Helper to find the row closest to a specific timestamp."""
        if df.empty or 'clock' not in df.columns:
            return None
        # Find index of minimum absolute difference
        idx = (df['clock'] - target_time).abs().idxmin()
        return df.loc[idx]
    
    def save_report(self, output_dir: Path, filename: str = "statistical_report.csv") -> None:
        """
        Saves the accumulated statistical results to a CSV file.
        Args:
            output_dir (Path): Directory to save the report.
            filename (str): Name of the output CSV file.
        """
        
        if not self.results_history:
            logger.warning("No statistical results to save.")
            return

        df_results = pd.DataFrame(self.results_history)
        df_results = self._apply_holm_correction(df_results)
        output_path = output_dir / filename
        df_results.to_csv(output_path, index=False)

        raw = int(df_results["is_significant"].sum()) if "is_significant" in df_results else 0
        adj = int(df_results["is_significant_holm"].sum()) if "is_significant_holm" in df_results else 0
        logger.info(
            f"💾 Statistical report saved to: {output_path.name} "
            f"({raw} significant raw, {adj} after Holm correction over "
            f"{len(df_results)} tests)"
        )

    def _apply_holm_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Holm-Bonferroni across every test in the report.

        The evaluator runs one t-test per discovered metric per competitor,
        which is ~60 tests for a four-arm plan. At alpha=0.05 that is about
        three false positives expected from chance alone, so the raw
        `is_significant` column overstates the result: on the 2026-09-12
        crash-check data it flagged 21 metrics, of which 8 survive correction -
        and those 8 are all cost/bandwidth, which is the claim the paper
        actually makes.

        Holm rather than plain Bonferroni: same family-wise error guarantee,
        uniformly more power, no extra assumptions. The raw column is kept so a
        reader can see both, and because pre-registering a small metric set is a
        legitimate alternative to correcting - but doing NEITHER is not.
        See thesis-hub/docs/decisions.md, 2026-09-12.
        """
        if "p_value" not in df.columns:
            return df

        df = df.copy()
        valid = df["p_value"].notna()
        m = int(valid.sum())
        df["n_tests_in_family"] = m
        df["p_value_holm"] = float("nan")
        df["is_significant_holm"] = False

        if m == 0:
            return df

        order = df.loc[valid, "p_value"].sort_values().index
        running_max = 0.0
        for rank, idx in enumerate(order):
            adjusted = min(1.0, (m - rank) * float(df.at[idx, "p_value"]))
            # Enforce monotonicity: an adjusted p may never fall below an
            # earlier (smaller raw p) one.
            running_max = max(running_max, adjusted)
            df.at[idx, "p_value_holm"] = running_max
            df.at[idx, "is_significant_holm"] = running_max < self.alpha

        return df