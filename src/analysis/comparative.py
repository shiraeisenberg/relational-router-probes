"""Comparative analysis across signals."""

from typing import Optional
import pandas as pd
from ..probing.linear_probe import ProbeResult


def compare_signals(results: list[ProbeResult]) -> pd.DataFrame:
    """Create comparison table from probe results."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


def generate_comparison_table(
    results: list[ProbeResult],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Generate and optionally save comparison table."""
    df = compare_signals(results)
    if output_path:
        df.to_csv(output_path, index=False)
    return df
