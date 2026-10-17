# src/analysis/aggregator.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class MetricsAggregator:
    """
    Aggregates simulation results across multiple random seeds.
    
    This class is responsible for loading CSV outputs from different simulation runs 
    (seeds) for a specific experimental variation, aligning them temporally, and 
    computing statistical measures (mean and standard deviation) to ensure 
    reproducibility and statistical significance.
    """

    def __init__(self, time_rounding_decimals: int = 1):
        """
        Args:
            time_rounding_decimals (int): Number of decimals to round the 'clock' column.
                                          Crucial for aligning discrete event simulation 
                                          timestamps across different asynchronous runs.
        """
        self.time_rounding_decimals = time_rounding_decimals

    def aggregate_variation(self, variation_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Aggregates all seed runs within a specific variation directory.

        Args:
            variation_dir (Path): Path to the variation folder (e.g., 'attack_lazy_30_percent').

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
                A tuple containing (aggregated_summary_df, aggregated_detailed_df).
                Returns None for missing data.
        """
        if not variation_dir.exists() or not variation_dir.is_dir():
            logger.warning(f"Variation directory not found: {variation_dir}")
            return None, None

        logger.info(f"Aggregating variation: {variation_dir.name}")

        seed_dirs = [d for d in variation_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        if not seed_dirs:
            logger.warning(f"No seed directories found in {variation_dir}")
            return None, None

        summary_df = self._process_csv_type(seed_dirs, "metrics_summary.csv", is_detailed=False)
        detailed_df = self._process_csv_type(seed_dirs, "metrics_nodes_detailed.csv", is_detailed=True)

        return summary_df, detailed_df

    def _process_csv_type(self, seed_dirs: List[Path], filename: str, is_detailed: bool) -> Optional[pd.DataFrame]:
        """
        Internal method to load, align, and compute statistics for a specific CSV type.
        """
        all_dfs = []
        for seed_dir in seed_dirs:
            csv_path = seed_dir / filename
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        # Extract seed number for tracking
                        seed_num = int(seed_dir.name.split('_')[1])
                        df['seed'] = seed_num
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to read {csv_path}: {e}")

        if not all_dfs:
            return None

        # Concatenate all seed DataFrames into one massive DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 💡 Architecture Note: Time Alignment
        #
        # Align on the ORDINAL reporting tick, not on the rounded wall clock.
        # Every seed emits the same number of reports at the same nominal
        # interval, but DES jitter moves the actual timestamps: in plan 23
        # group 3, seed 42 reported at t=390.5 while seeds 43/44 reported at
        # t=390.6. Rounding to `time_rounding_decimals` (0.1s) put those in
        # DIFFERENT buckets, so a 3-seed run aggregated into 32 rows of which
        # 23 held a single seed, 8 held two, and exactly ONE held all three.
        #
        # The damage was silent and total: mean lines zig-zagged between
        # different seed subsets (a 14-point "drop" between t=390.5 and
        # t=390.6 was purely which seeds landed in which bucket), `std` was
        # NaN wherever count==1, and StatEngine rejected almost every metric
        # with "Insufficient sample size (N<2)" - the saved report came back
        # with empty p-values. Widening the rounding would only trade one
        # arbitrary bucket boundary for another; the tick index is exact.
        # See thesis-hub/docs/decisions.md, 2026-09-12.
        combined_df = combined_df.sort_values(['seed', 'clock'], kind='mergesort')
        tick_keys = ['seed', 'node_id'] if is_detailed else ['seed']
        combined_df['tick'] = combined_df.groupby(tick_keys, sort=False).cumcount()

        # Keep the real clock as a reportable quantity (averaged per tick) so
        # plots still carry simulation time on the x-axis.
        combined_df['clock_aligned'] = combined_df.groupby(
            ['tick', 'node_id'] if is_detailed else ['tick']
        )['clock'].transform('mean').round(self.time_rounding_decimals)

        # Define the grouping keys
        groupby_cols = ['clock_aligned', 'node_id'] if is_detailed else ['clock_aligned']

        # Select only numeric columns for mean/std calculations (drop string columns like node_type)
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        # 'tick' is an alignment key, not a measurement - excluded like 'seed'/'clock'.
        cols_to_agg = [
            col for col in numeric_cols
            if col not in groupby_cols and col not in ('seed', 'clock', 'tick')
        ]

        # Vectorized Pandas Aggregation (Highly Optimized)
        aggregated = combined_df.groupby(groupby_cols)[cols_to_agg].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten the MultiIndex columns (e.g., 'global_mean_accuracy_mean', 'global_mean_accuracy_std')
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        
        # Rename 'clock_aligned' back to 'clock' for downstream consistency
        aggregated.rename(columns={'clock_aligned': 'clock'}, inplace=True)

        logger.debug(f"Successfully aggregated {filename} across {len(all_dfs)} seeds.")
        return aggregated