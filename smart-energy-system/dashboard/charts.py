"""Plotly chart builders with one consistent, accessible visual style.

Style rules: thin 2px lines, one y-axis per chart (no dual axes), recessive hairline grid,
fixed colour roles (actual = blue, predicted = orange, peaks/threshold = critical red),
single-hue blue ramp for magnitude heatmaps, hover tooltips everywhere.
"""

import pandas as pd
import plotly.graph_objects as go

# Colour roles (validated reference palette).
ACTUAL = "#2a78d6"  # categorical slot 1
PREDICTED = "#eb6834"  # categorical slot 2
SECONDARY = "#1baf7a"  # categorical slot 3
CRITICAL = "#d03b3b"  # status: critical (peaks / threshold only)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def with_hourly_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Re-insert missing hours as NaN so line charts show a GAP instead of a straight line
    across hours with no data (missing data must not look like real readings)."""
    if df.empty:
        return df
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    return df.set_index("timestamp").reindex(full).rename_axis("timestamp").reset_index()


def _threshold_line(fig: go.Figure, threshold: float) -> None:
    fig.add_hline(
        y=threshold,
        line={"color": CRITICAL, "width": 1},
        annotation_position="top left",
        annotation_text=f"Peak threshold {threshold:.2f} kWh",
        annotation_font_color=INK_2,
        annotation_bgcolor=SURFACE,
    )


def _style(fig: go.Figure, y_title: str = "", x_title: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK_2, "size": 13},
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font": {"color": INK}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        bargap=0.15,
    )
    axis = {"gridcolor": GRID, "linecolor": BASELINE, "zeroline": False, "tickfont": {"color": MUTED}}
    fig.update_xaxes(title=x_title, **axis)
    fig.update_yaxes(title=y_title, rangemode="tozero", **axis)
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, name: str, y_title: str, color: str = ACTUAL) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df[x],
            y=df[y],
            name=name,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
        )
    )
    return _style(fig, y_title)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    y_title: str,
    color: str = ACTUAL,
    hover_fmt: str = ".2f",
    horizontal: bool = False,
    hovermode: str = "closest",
) -> go.Figure:
    bar = (
        go.Bar(
            y=df[x],
            x=df[y],
            orientation="h",
            marker={"color": color, "cornerradius": 4},
            hovertemplate="%{y}: %{x:" + hover_fmt + "}<extra></extra>",
        )
        if horizontal
        else go.Bar(
            x=df[x],
            y=df[y],
            marker={"color": color, "cornerradius": 4},
            hovertemplate="%{x}: %{y:" + hover_fmt + "}<extra></extra>",
        )
    )
    fig = _style(go.Figure(bar), "" if horizontal else y_title)
    if horizontal:
        fig.update_xaxes(title=y_title)
        fig.update_yaxes(autorange="reversed", rangemode="normal")
    fig.update_layout(hovermode=hovermode)
    return fig


def consumption_with_peaks(df: pd.DataFrame, threshold: float | None) -> go.Figure:
    """Hourly consumption line, peak hours marked, threshold as a reference line."""
    fig = go.Figure(
        go.Scatter(
            x=df["timestamp"],
            y=df["consumption_kwh"],
            name="Consumption",
            mode="lines",
            line={"color": ACTUAL, "width": 1.5},
            hovertemplate="%{y:.3f} kWh<extra>Consumption</extra>",
        )
    )
    if threshold is not None:
        peaks = df.loc[df["consumption_kwh"] > threshold]
        fig.add_trace(
            go.Scatter(
                x=peaks["timestamp"],
                y=peaks["consumption_kwh"],
                name="Peak hour",
                mode="markers",
                marker={"color": CRITICAL, "size": 8, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="%{y:.3f} kWh<extra>Peak hour</extra>",
            )
        )
        _threshold_line(fig, threshold)
    return _style(fig, "kWh per hour")


def actual_vs_predicted(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["actual_kwh"],
            name="Actual",
            mode="lines",
            line={"color": ACTUAL, "width": 2},
            hovertemplate="%{y:.3f} kWh<extra>Actual</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_kwh"],
            name="Predicted",
            mode="lines",
            line={"color": PREDICTED, "width": 2},
            hovertemplate="%{y:.3f} kWh<extra>Predicted</extra>",
        )
    )
    return _style(fig, "kWh per hour")


def error_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df["timestamp"],
            y=df["error_kwh"],
            name="Error",
            marker={"color": PREDICTED},
            hovertemplate="%{y:+.3f} kWh<extra>Predicted − actual</extra>",
        )
    )
    fig = _style(fig, "Predicted − actual (kWh)")
    fig.update_yaxes(rangemode="normal", zeroline=True, zerolinecolor=BASELINE)
    return fig


def predicted_vs_actual_scatter(df: pd.DataFrame) -> go.Figure:
    upper = float(max(df["actual_kwh"].max(), df["predicted_kwh"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, upper],
            y=[0, upper],
            mode="lines",
            name="Perfect prediction",
            line={"color": MUTED, "width": 1},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["actual_kwh"],
            y=df["predicted_kwh"],
            mode="markers",
            name="Hours",
            marker={"color": ACTUAL, "size": 6, "opacity": 0.5},
            hovertemplate="actual %{x:.3f} · predicted %{y:.3f}<extra></extra>",
        )
    )
    fig = _style(fig, "Predicted kWh", "Actual kWh")
    fig.update_layout(hovermode="closest")
    return fig


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"],
            y=history["consumption_kwh"],
            name="Recent actual",
            mode="lines",
            line={"color": ACTUAL, "width": 2},
            hovertemplate="%{y:.3f} kWh<extra>Actual</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["predicted_kwh"],
            name="Forecast",
            mode="lines+markers",
            line={"color": PREDICTED, "width": 2},
            marker={"size": 8, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y:.3f} kWh<extra>Forecast</extra>",
        )
    )
    _threshold_line(fig, threshold)
    return _style(fig, "kWh per hour")


def weekday_hour_heatmap(df: pd.DataFrame) -> go.Figure:
    """Average consumption by day-of-week × hour-of-day (single-hue sequential scale)."""
    pivot = (
        df.assign(dow=df["timestamp"].dt.dayofweek, hour=df["timestamp"].dt.hour)
        .pivot_table(index="dow", columns="hour", values="consumption_kwh", aggfunc="mean")
        .reindex(index=range(7), columns=range(24))
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[f"{h:02d}:00" for h in range(24)],
            y=WEEKDAYS,
            colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
            colorbar={"title": "kWh/h"},
            xgap=2,
            ygap=2,
            hovertemplate="%{y} %{x}: %{z:.3f} kWh<extra></extra>",
        )
    )
    fig = _style(fig, "", "Hour of day", height=300)
    fig.update_yaxes(autorange="reversed", rangemode="normal", gridcolor=SURFACE)
    fig.update_xaxes(gridcolor=SURFACE)
    fig.update_layout(hovermode="closest")
    return fig
