from typing import List, Optional, Tuple

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
    """Plot two binary event series as vertical tick marks on parallel baselines.

    Parameters
    ----------
    df:
        DataFrame whose columns are ``(lon, lat)`` coordinate tuples and whose
        index is a regular datetime range.
    cell_1_coords, cell_2_coords:
        Column keys identifying the two cells to compare.
    event_color_1, event_color_2:
        Colours for the tick marks of each series.
    baseline_color:
        Colour of the horizontal baseline drawn for each series.
    bar_height:
        Height of each event tick mark in plot units.
    gap:
        Vertical separation between the two baselines in plot units.
    """
    if cell_1_coords not in df.columns:
        raise ValueError(f"Cell {cell_1_coords} not found in the dataset.")
    if cell_2_coords not in df.columns:
        raise ValueError(f"Cell {cell_2_coords} not found in the dataset.")

    dates = df.index
    series_1 = df[cell_1_coords]
    series_2 = df[cell_2_coords]

    fig = go.Figure()

    for y_base, series, color in [
        (0, series_1, event_color_1),
        (-gap, series_2, event_color_2),
    ]:
        fig.add_trace(go.Scatter(
            x=dates,
            y=[y_base] * len(dates),
            mode="lines",
            line=dict(color=baseline_color, width=line_width),
            hoverinfo="skip",
        ))

        # Build all event ticks as a single trace with None separators between
        # segments; much faster than one trace per event for long series.
        event_x: List[Optional[pd.Timestamp]] = []
        event_y: List[Optional[float]] = []
        for i, value in enumerate(series):
            if value == 1:
                event_x += [dates[i], dates[i], None]
                event_y += [y_base, y_base + bar_height, None]
        if event_x:
            fig.add_trace(go.Scatter(
                x=event_x,
                y=event_y,
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
        template="plotly_white",
    )
    return fig
