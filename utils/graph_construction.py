from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from plotly_resampler import FigureResampler


# --------- graph construction logic + callback ---------
def visualize_multiple_files(file_list: List[Union[str, Path]]) -> FigureResampler:
    """Create FigureResampler where each subplot row represents all signals from a file.

    Parameters
    ----------
    file_list: List[Union[str, Path]]

    Returns
    -------
    FigureResampler
        Returns a view of the existing, global FigureResampler object.

    """
    fig = FigureResampler(make_subplots(rows=len(file_list), shared_xaxes=False))
    fig.update_layout(height=min(900, 350 * len(file_list)))

    for i, f in enumerate(file_list, 1):
        df = pl.read_parquet(f)  # TODO: replace with more generic data loading code
        if "Timestamp" in df.columns:
            df = df.sort("Timestamp")

        for c in df.columns[::-1]:
            fig.add_trace(
                go.Scattergl(
                    name=c,
                    x=df["Timestamp"].to_numpy().tolist(),
                    y=df[c].to_numpy().tolist(),
                    dx=i,
                    dy=1,
                )
            )
            # fig.add_trace(go.Scattergl(name=c), hf_x=df.index, hf_y=df[d], row=i, col=2)
    return fig
