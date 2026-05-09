from typing import Tuple

import pandas as pd
import plotly.graph_objects as go


def visualize_two_series_events(
    df: pd.DataFrame,
    cell_1_coords: Tuple[float, float],
    cell_2_coords: Tuple[float, float],
    event_color_1: str = "blue",
    event_color_2: str = "red",
    baseline_color: str = "green",
    title: str = "Comparison of Two Event Series",
    line_width: int = 2,
    bar_height: int = 8,
    gap: int = 15,
) -> go.Figure:
    if cell_1_coords not in df.columns:
        raise ValueError(f"Cell {cell_1_coords} not found in the dataset.")
    if cell_2_coords not in df.columns:
        raise ValueError(f"Cell {cell_2_coords} not found in the dataset.")

    series_1 = df[cell_1_coords]
    series_2 = df[cell_2_coords]
    dates = df.index

    fig = go.Figure()

    for y_base, series, color in [
        (0, series_1, event_color_1),
        (-gap, series_2, event_color_2),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=[y_base] * len(dates),
            mode="lines",
            line=dict(color=baseline_color, width=line_width),
            hoverinfo="skip",
        ))
        for i, value in enumerate(series):
            if value == 1:
                fig.add_trace(go.Scatter(
                    x=[dates[i], dates[i]],
                    y=[y_base, y_base + bar_height],
                    mode="lines",
                    line=dict(color=color, width=3),
                    hovertemplate=f"Date: {dates[i]}<extra></extra>",
                ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
        template="plotly_white",
    )
    return fig
