# src/analysis/plotter.py
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _strip_mean_suffix(column: str) -> str:
    """
    Drops a trailing '_mean' from an aggregated column name.

    Must not be done with `str.replace('_mean', '')`, which removes EVERY
    occurrence. The headline metric of this project is `global_mean_accuracy`,
    so its aggregated column is `global_mean_accuracy_mean` and replace() turned
    it into `global_accuracy` -- a column that does not exist. The variance band
    was therefore silently skipped for exactly the metric the paper leads with,
    and StatEngine (which had the same bug) rejected it as having no variance
    data. Metrics with a single '_mean', like `global_max_accuracy_mean`, were
    unaffected, which is why the failure looked sporadic rather than systematic.
    See thesis-hub/docs/decisions.md, 2026-09-12.
    """
    return column[: -len("_mean")] if column.endswith("_mean") else column

class ResultsPlotter:
    """
    Generates publication-ready plots conforming to IEEE and Springer standards.
    
    This class applies a consistent scientific styling to all generated figures,
    ensuring high-resolution outputs suitable for LaTeX inclusion.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initializes the plotter and sets up the output directory and styling.
        
        Args:
            output_dir (Path): The directory where generated PDF plots will be saved.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_ieee_standards()

    def _apply_ieee_standards(self) -> None:
        """
        Configures matplotlib and seaborn parameters to match standard IEEE guidelines.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        
        plt.rcParams.update({
            'font.family': 'serif',
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 2.5,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.format': 'pdf'
        })

    #: Per-series visual encoding, cycled by series index.
    #:
    #: Colour alone is not a usable channel here: these figures go into papers
    #: that are printed and photocopied in greyscale, and readers with colour
    #: vision deficiency see two of the default matplotlib colours as nearly the
    #: same. Each series therefore carries two independent non-colour cues - a
    #: dash pattern and a marker glyph - either of which identifies it on its
    #: own. Verified by rasterising a four-series figure to greyscale.
    #:
    #: Markers are drawn unfilled and sparsely (`markevery`) so they identify a
    #: curve without hiding the data or colliding on dense time series.
    #: Variance is shown as discrete error bars (see plot_time_series_with_
    #: variance), not a shaded band, so no hatch cue is needed here.
    _LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]
    _MARKERS = ["o", "s", "^", "D", "v", "P"]

    def _series_style(self, index: int) -> Dict[str, Any]:
        """Returns the dash/marker style pair for the n-th series."""
        return {
            "linestyle": self._LINESTYLES[index % len(self._LINESTYLES)],
            "marker": self._MARKERS[index % len(self._MARKERS)],
            # Offset the marker phase per series so markers from different
            # curves do not stack on the same x positions where curves overlap.
            "markevery": (index % 4, 4),
        }

    def plot_time_series_with_variance(
        self,
        data_dict: Dict[str, pd.DataFrame],
        metric: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        x_col: str = 'clock'
    ) -> None:
        """
        Plots a time-series metric comparing multiple protocols, with a shaded
        band of +/- one standard deviation across seeds.

        The band is one SD, NOT a confidence interval -- the docstring used to
        say "confidence intervals", and a paper that reproduces that wording
        would be overstating precision: at n=3 seeds a 95% CI is roughly
        +/- 2.5 SD, so the shaded region is far narrower than a CI would be.
        The legend labels it explicitly so the figure cannot be misread.

        Args:
            data_dict (Dict[str, pd.DataFrame]): Mapping of protocol names to their respective DataFrames.
            metric (str): The base metric name to plot (e.g., 'global_mean_accuracy').
            title (str): The title of the plot.
            xlabel (str): The label for the X-axis.
            ylabel (str): The label for the Y-axis.
            filename (str): The output filename without extension.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine standard column naming conventions dynamically
        first_df = next(iter(data_dict.values()))
        mean_col = metric if metric in first_df.columns else f"{metric}_mean"
        std_col = f"{_strip_mean_suffix(mean_col)}_std"

        for idx, (label, df) in enumerate(data_dict.items()):
            if df is None or df.empty or mean_col not in df.columns:
                logger.warning(f"Skipping plot for '{label}': Metric '{mean_col}' not found.")
                continue

            time_series = df[x_col]
            mean_series = df[mean_col]
            style = self._series_style(idx)

            # Plot the primary mean line
            line, = ax.plot(
                time_series, mean_series, label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=style["markevery"],
                markerfacecolor="none",
                markersize=7,
            )

            # Standard deviation as discrete error bars, not a shaded band.
            #
            # 2026-09-13 (thesis-hub/docs/decisions.md, same date): a
            # translucent fill_between band - even a neutral-grey one tried
            # first - still reads as a halo/glow once several series
            # overlap on the same axes. Error bars carry the same "here is
            # the spread" information as discrete marks at each observed
            # point, with no continuous shaded region at all. Drawn at a
            # sparser cadence than the line's own markers (every 2nd marker
            # position) so dense time series do not turn into a solid mass
            # of caps.
            if std_col in df.columns and not df[std_col].isna().all():
                std_series = df[std_col]
                start, step = style["markevery"]
                err_idx = list(range(start, len(time_series), step * 2))
                if err_idx:
                    ax.errorbar(
                        time_series.iloc[err_idx],
                        mean_series.iloc[err_idx],
                        yerr=std_series.iloc[err_idx],
                        fmt="none",
                        ecolor="0.35",
                        elinewidth=0.8,
                        capsize=3,
                        capthick=0.8,
                    )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # State what the shading is, and on how many seeds, directly on the figure.
        count_col = f"{_strip_mean_suffix(mean_col)}_count"
        seed_counts = [
            int(df[count_col].max())
            for df in data_dict.values()
            if df is not None and not df.empty and count_col in df.columns
        ]
        if seed_counts:
            n = min(seed_counts)
            ax.legend(title=f"error bars: $\\pm$1 SD over {n} seed{'s' if n != 1 else ''}")
        else:
            ax.legend()
        
        output_path = self.output_dir / f"{filename}.pdf"
        
        try:
            fig.savefig(output_path)
            logger.info(f"Successfully generated plot: {output_path.name}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close(fig)