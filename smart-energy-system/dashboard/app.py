"""Streamlit dashboard (Phase 17).

Run:  streamlit run dashboard/app.py      (the API must be running — see README)

All data comes from the FastAPI backend through ``ApiClient``; the dashboard holds no business
logic of its own beyond formatting and charting.
"""

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `import dashboard.*`

from dashboard import charts  # noqa: E402
from dashboard.api_client import ApiClient, ApiError  # noqa: E402

st.set_page_config(page_title="Smart Energy Dashboard", page_icon="⚡", layout="wide")
api = ApiClient()
ALL_METERS = "All meters"


# ------------------------------------------------------------------ cached API calls
@st.cache_data(ttl=120, show_spinner=False)
def fetch(path: str, **params):
    return api.get(path, **params)


def to_frame(records, time_cols=("timestamp",)) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in time_cols:
        if col in df:
            df[col] = pd.to_datetime(df[col])
    return df


def fmt_money(value: float, currency: str) -> str:
    return f"{currency} {value:,.0f}" if abs(value) >= 1000 else f"{currency} {value:,.2f}"


# ------------------------------------------------------------------ connection check
try:
    health = fetch("/health")
    meters = fetch("/meters")
    settings = fetch("/settings")
except ApiError as exc:
    st.error(f"**Backend unavailable.** {exc}")
    st.stop()

if not meters:
    st.warning("The database has no meters yet. Run `python -m scripts.load_to_db`.")
    st.stop()

meters_df = to_frame(meters, ("first_timestamp", "last_timestamp"))

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚡ Smart Energy")
    meter_choice = st.selectbox("Meter", [ALL_METERS, *meters_df["meter_code"]])
    meter_code = None if meter_choice == ALL_METERS else meter_choice

    scope = meters_df if meter_code is None else meters_df.loc[meters_df["meter_code"] == meter_code]
    data_start = scope["first_timestamp"].min().date()
    data_end = scope["last_timestamp"].max().date()
    default_start = max(data_start, data_end - timedelta(days=30))
    picked = st.date_input("Date range", value=(default_start, data_end), min_value=data_start, max_value=data_end)
    start_date, end_date = picked if isinstance(picked, tuple) and len(picked) == 2 else (default_start, data_end)
    start = datetime.combine(start_date, time.min).isoformat()
    end = datetime.combine(end_date, time(23, 59, 59)).isoformat()

    st.divider()
    st.subheader("Settings")
    with st.form("settings"):
        tariff = st.number_input(
            f"Tariff ({settings['currency']}/kWh)",
            min_value=0.0,
            max_value=1000.0,
            value=float(settings["tariff_per_kwh"]),
            step=0.5,
            help="Flat tariff used for ESTIMATED cost. Not taken from a real bill.",
        )
        percentile = st.slider(
            "Peak threshold (percentile)",
            min_value=50.0,
            max_value=99.0,
            value=float(settings["peak_percentile"]),
            step=1.0,
            help="An hour is a peak if consumption is above this percentile of the meter's history.",
        )
        if st.form_submit_button("Save settings", use_container_width=True):
            try:
                api.put("/settings", {"tariff_per_kwh": tariff, "peak_percentile": percentile})
                st.cache_data.clear()
                st.success("Settings saved")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))

    st.divider()
    model_label = (
        f"{health.get('model_name')} ({health.get('model_version')})" if health.get("model_loaded") else "not loaded"
    )
    st.caption(f"Model: {model_label}  \nData source: {health.get('data_source')}  \nAPI: {api.base_url}")

# ------------------------------------------------------------------ header
st.title("Smart Energy Consumption Dashboard")
st.caption(
    f"{meter_choice} · {start_date:%d %b %Y} → {end_date:%d %b %Y} · "
    f"Timezone {settings['timezone']} · Peak = above P{settings['peak_percentile']:.0f}"
)
if health.get("is_synthetic"):
    st.warning(
        "**Synthetic demo data.** The model and charts are built from generated data with the CEEW "
        "schema, not the real Bareilly dataset. Re-run the pipeline on the real CSV before using results."
    )

try:
    overview = fetch("/overview", meter_code=meter_code, start=start, end=end)
except ApiError as exc:
    st.error(str(exc))
    st.stop()
currency = overview["currency"]

tabs = st.tabs(
    ["Overview", "Consumption", "Prediction", "Peak analysis", "Cost", "Recommendations", "Model performance"]
)

# ================================================================== OVERVIEW
with tabs[0]:
    next_hour = None
    if meter_code and health.get("model_loaded"):
        try:
            next_hour = api.post("/predict", {"meter_code": meter_code, "horizon_hours": 1, "store": False})
        except ApiError:
            next_hour = None

    row1, row2 = st.columns(3), st.columns(3)
    row1[0].metric("Total consumption", f"{overview['total_kwh']:,.0f} kWh")
    row1[1].metric("Average per hour", f"{overview['average_hourly_kwh']:.3f} kWh")
    row1[2].metric(
        "Highest hour",
        f"{overview['peak_hourly_kwh']:.2f} kWh" if overview["peak_hourly_kwh"] else "—",
        help=f"At {overview['peak_timestamp']}" if overview["peak_timestamp"] else None,
    )
    row2[0].metric(
        overview["cost_label"], fmt_money(overview["estimated_cost"], currency), help=overview["cost_disclaimer"]
    )
    row2[1].metric("Peak hours", f"{overview['peak_event_count']:,}")
    row2[2].metric(
        "Predicted next hour",
        f"{next_hour['forecast'][0]['predicted_kwh']:.3f} kWh" if next_hour else "—",
        help=(
            f"For {next_hour['forecast'][0]['timestamp']} (hour after latest data)"
            if next_hour
            else "Select a single meter to see its next-hour prediction"
        ),
    )

    daily = to_frame(fetch("/consumption", meter_code=meter_code, start=start, end=end, granularity="daily")["points"])
    st.subheader("Daily consumption")
    if daily.empty:
        st.info("No data in the selected range.")
    else:
        st.plotly_chart(
            charts.line_chart(daily, "timestamp", "consumption_kwh", "Daily kWh", "kWh per day"),
            use_container_width=True,
        )
        st.caption("Days with missing hours (meter offline) show lower totals — missing hours are not filled in.")

    if meter_code is None:
        by_meter = pd.DataFrame(fetch("/consumption/by-meter", start=start, end=end))
        if not by_meter.empty:
            st.subheader("Consumption by meter")
            st.plotly_chart(
                charts.bar_chart(
                    by_meter, "meter_code", "total_kwh", "Total kWh", horizontal=True, hover_fmt=",.1f"
                ).update_layout(height=max(300, 22 * len(by_meter))),
                use_container_width=True,
            )

# ================================================================== CONSUMPTION
with tabs[1]:
    granularity = st.radio("Granularity", ["hourly", "daily", "weekly", "monthly"], horizontal=True)
    series = to_frame(
        fetch("/consumption", meter_code=meter_code, start=start, end=end, granularity=granularity)["points"]
    )
    if series.empty:
        st.info("No data in the selected range.")
    else:
        unit = {"hourly": "kWh per hour", "daily": "kWh per day", "weekly": "kWh per week", "monthly": "kWh per month"}[
            granularity
        ]
        st.subheader(f"{granularity.capitalize()} consumption trend")
        st.plotly_chart(
            charts.line_chart(
                charts.with_hourly_gaps(series) if granularity == "hourly" else series,
                "timestamp",
                "consumption_kwh",
                "Consumption",
                unit,
            ),
            use_container_width=True,
        )

        hourly = to_frame(
            fetch("/consumption", meter_code=meter_code, start=start, end=end, granularity="hourly")["points"]
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Average by weekday × hour")
            st.plotly_chart(charts.weekday_hour_heatmap(hourly), use_container_width=True)
        with right:
            st.subheader("Weekday vs weekend profile")
            profile = (
                hourly.assign(
                    hour=hourly["timestamp"].dt.hour,
                    day_type=hourly["timestamp"].dt.dayofweek.map(lambda d: "Weekend" if d >= 5 else "Weekday"),
                )
                .groupby(["day_type", "hour"])["consumption_kwh"]
                .mean()
                .unstack(0)
                .reset_index()
            )
            fig = charts.line_chart(profile, "hour", "Weekday", "Weekday", "Average kWh per hour")
            if "Weekend" in profile:
                fig.add_scatter(
                    x=profile["hour"],
                    y=profile["Weekend"],
                    name="Weekend",
                    mode="lines",
                    line={"color": charts.PREDICTED, "width": 2},
                )
            fig.update_xaxes(title="Hour of day", dtick=3)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average voltage")
        st.caption("Shown on its own chart (different unit from consumption).")
        st.plotly_chart(
            charts.line_chart(
                series.dropna(subset=["avg_voltage"]),
                "timestamp",
                "avg_voltage",
                "Voltage",
                "Volts",
                color=charts.SECONDARY,
            ).update_yaxes(rangemode="normal"),
            use_container_width=True,
        )
        with st.expander("Data table"):
            st.dataframe(series, use_container_width=True, hide_index=True)

# ================================================================== PREDICTION
with tabs[2]:
    if meter_code is None:
        st.info("Select a single meter in the sidebar to see predictions.")
    else:
        st.subheader("Forecast")
        col_a, col_b, col_c = st.columns([2, 1, 1])
        horizon = col_a.slider(
            "Hours ahead",
            1,
            24,
            6,
            help="1 hour = the evaluated use-case. Longer horizons are recursive and less accurate.",
        )
        store = col_b.checkbox("Save to database", value=False)
        run = col_c.button("Run forecast", type="primary", use_container_width=True)
        if run:
            try:
                st.session_state["forecast"] = api.post(
                    "/predict", {"meter_code": meter_code, "horizon_hours": horizon, "store": store}
                )
            except ApiError as exc:
                st.error(str(exc))

        result = st.session_state.get("forecast")
        if result and result["meter_code"] == meter_code:
            fc = to_frame(result["forecast"])
            until = pd.Timestamp(result["based_on_data_until"])
            recent = to_frame(
                fetch(
                    "/consumption",
                    meter_code=meter_code,
                    start=(until - pd.Timedelta(hours=72)).isoformat(),
                    end=until.isoformat(),
                )["points"]
            )
            k = st.columns(3)
            k[0].metric("Forecast total", f"{result['total_predicted_kwh']:.2f} kWh")
            k[1].metric(result["cost_label"], fmt_money(result["total_estimated_cost"], result["currency"]))
            k[2].metric("Predicted peak hours", int(fc["is_peak"].sum()))
            st.plotly_chart(
                charts.forecast_chart(charts.with_hourly_gaps(recent), fc, result["peak_threshold_kwh"]),
                use_container_width=True,
            )
            for note in result["notes"]:
                st.caption(f"ℹ️ {note}")
            with st.expander("Forecast table"):
                st.dataframe(fc, use_container_width=True, hide_index=True)
        else:
            st.caption("Choose a horizon and press **Run forecast**.")

        st.divider()
        st.subheader("Actual vs predicted (test period back-test)")
        st.caption("Next-hour predictions made by the saved model on the held-out test period it never trained on.")
        backtest = to_frame(
            fetch("/predictions", meter_code=meter_code, prediction_type="backtest", start=start, end=end)
        )
        if backtest.empty:
            backtest_all = to_frame(fetch("/predictions", meter_code=meter_code, prediction_type="backtest"))
            if backtest_all.empty:
                st.info("No back-test predictions stored for this meter.")
            else:
                st.info(
                    f"The test period is {backtest_all['timestamp'].min():%d %b %Y} → "
                    f"{backtest_all['timestamp'].max():%d %b %Y}. Widen the date range to include it."
                )
        else:
            valid = backtest.dropna(subset=["actual_kwh"])
            mae = (valid["error_kwh"]).abs().mean()
            rmse = (valid["error_kwh"] ** 2).mean() ** 0.5
            m = st.columns(3)
            m[0].metric("MAE (this meter, range)", f"{mae:.4f} kWh")
            m[1].metric("RMSE (this meter, range)", f"{rmse:.4f} kWh")
            m[2].metric("Hours compared", f"{len(valid):,}")
            st.plotly_chart(charts.actual_vs_predicted(valid), use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.subheader("Prediction error over time")
                st.plotly_chart(charts.error_chart(valid), use_container_width=True)
            with right:
                st.subheader("Predicted vs actual")
                st.plotly_chart(charts.predicted_vs_actual_scatter(valid), use_container_width=True)

# ================================================================== PEAKS
with tabs[3]:
    summary = fetch("/peaks/summary", meter_code=meter_code, start=start, end=end)
    k = st.columns(4)
    k[0].metric("Peak hours", f"{summary['peak_hours']:,}")
    k[1].metric("Peak frequency", f"{summary['peak_frequency_pct']:.1f}% of hours")
    k[2].metric("Highest peak", f"{summary['max_kwh']:.2f} kWh" if summary["max_kwh"] else "—")
    k[3].metric(
        "Most common peak hours",
        " · ".join(f"{h:02d}h" for h in sorted(summary["most_common_peak_hours"])) or "—",
    )
    st.caption(
        f"Peak = hourly consumption above the meter's P{summary['percentile']:.0f} "
        "(computed per meter from its full history). Change the percentile in the sidebar."
    )

    if meter_code:
        threshold = summary["thresholds"].get(meter_code)
        hourly = to_frame(
            fetch("/consumption", meter_code=meter_code, start=start, end=end, granularity="hourly")["points"]
        )
        if not hourly.empty:
            st.subheader("Hourly consumption with peak hours")
            st.plotly_chart(
                charts.consumption_with_peaks(charts.with_hourly_gaps(hourly), threshold), use_container_width=True
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Peak hours by hour of day")
        by_hour = pd.DataFrame(
            {
                "hour": [f"{int(h):02d}:00" for h in summary["by_hour_of_day"]],
                "count": list(summary["by_hour_of_day"].values()),
            }
        )
        st.plotly_chart(
            charts.bar_chart(by_hour, "hour", "count", "Peak hours", color=charts.CRITICAL, hover_fmt="d"),
            use_container_width=True,
        )
    with right:
        st.subheader("Peak hours by day of week")
        by_day = pd.DataFrame({"day": charts.WEEKDAYS, "count": list(summary["by_day_of_week"].values())})
        st.plotly_chart(
            charts.bar_chart(by_day, "day", "count", "Peak hours", color=charts.CRITICAL, hover_fmt="d"),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Dates with most peak hours")
        st.dataframe(pd.DataFrame(summary["top_dates"]), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Largest peak events")
        events = pd.DataFrame(fetch("/peaks", meter_code=meter_code, start=start, end=end, order_by="kwh", limit=20))
        st.dataframe(events, use_container_width=True, hide_index=True)

# ================================================================== COST
with tabs[4]:
    st.warning(f"**{overview['cost_label']}** — {overview['cost_disclaimer']}", icon="ℹ️")
    rate = overview["tariff_per_kwh"]
    k = st.columns(3)
    k[0].metric(overview["cost_label"], fmt_money(overview["estimated_cost"], currency))
    k[1].metric("Tariff used", f"{currency} {rate:.2f} / kWh")
    days = max((end_date - start_date).days + 1, 1)
    k[2].metric("Average per day", fmt_money(overview["estimated_cost"] / days, currency))

    period = st.radio("Period", ["daily", "weekly", "monthly"], horizontal=True, key="cost_period")
    cost = to_frame(fetch("/consumption", meter_code=meter_code, start=start, end=end, granularity=period)["points"])
    if not cost.empty:
        cost["estimated_cost"] = cost["consumption_kwh"] * rate
        st.subheader(f"{period.capitalize()} estimated cost")
        st.plotly_chart(
            charts.bar_chart(
                cost,
                "timestamp",
                "estimated_cost",
                f"Estimated cost ({currency})",
                hover_fmt=",.2f",
                hovermode="x unified",
            ),
            use_container_width=True,
        )
        with st.expander("Cost table"):
            st.dataframe(
                cost[["timestamp", "consumption_kwh", "estimated_cost"]], use_container_width=True, hide_index=True
            )

    if meter_code is None:
        by_meter = pd.DataFrame(fetch("/consumption/by-meter", start=start, end=end))
        if not by_meter.empty:
            by_meter["estimated_cost"] = by_meter["total_kwh"] * rate
            st.subheader("Estimated cost by meter")
            st.plotly_chart(
                charts.bar_chart(
                    by_meter,
                    "meter_code",
                    "estimated_cost",
                    f"Estimated cost ({currency})",
                    horizontal=True,
                    hover_fmt=",.2f",
                ).update_layout(height=max(300, 22 * len(by_meter))),
                use_container_width=True,
            )

# ================================================================== RECOMMENDATIONS
with tabs[5]:
    st.caption(
        "Rule-based advice generated from predictions, peak periods and historical patterns. "
        "The system only recommends — it does not control any appliance."
    )
    recs = fetch("/recommendations", meter_code=meter_code, limit=30)
    if not recs:
        st.info("No recommendations yet. Run a forecast on the Prediction tab to generate new ones.")
    show = {"critical": st.error, "warning": st.warning, "info": st.info}
    icons = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
    for rec in recs:
        saving = (
            f"  \n*Potential saving (estimate): {rec['potential_saving_kwh']:.2f} kWh*"
            if rec.get("potential_saving_kwh")
            else ""
        )
        label = rec["recommendation_type"].replace("_", " ").capitalize()
        show[rec["severity"]](
            f"**{rec['severity'].upper()} · {label}**  \n{rec['message']}{saving}", icon=icons[rec["severity"]]
        )

# ================================================================== MODEL PERFORMANCE
with tabs[6]:
    perf = fetch("/model-performance")
    models = pd.DataFrame(perf["models"])
    if models.empty:
        st.info("No model results stored.")
    else:
        st.caption(
            f"Evaluated on the {perf['evaluation_set']}. Best model chosen by {perf['selection_metric']}. "
            f"Data: {perf['data_source']}{' (SYNTHETIC)' if perf['is_synthetic'] else ''}."
        )
        table = models[["model_name", "mae", "rmse", "r2", "nmae_pct", "is_active"]].rename(
            columns={
                "model_name": "Model",
                "mae": "MAE (kWh)",
                "rmse": "RMSE (kWh)",
                "r2": "R²",
                "nmae_pct": "NMAE (%)",
                "is_active": "Selected",
            }
        )
        st.subheader("Model comparison")
        st.dataframe(
            table.style.format({"MAE (kWh)": "{:.4f}", "RMSE (kWh)": "{:.4f}", "R²": "{:.4f}", "NMAE (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        left, right = st.columns(2)
        with left:
            st.subheader("MAE by model (lower is better)")
            st.plotly_chart(
                charts.bar_chart(
                    models, "model_name", "mae", "MAE (kWh)", horizontal=True, hover_fmt=".4f"
                ).update_layout(height=260),
                use_container_width=True,
            )
        with right:
            st.subheader("R² by model (higher is better)")
            st.plotly_chart(
                charts.bar_chart(models, "model_name", "r2", "R²", horizontal=True, hover_fmt=".4f").update_layout(
                    height=260
                ),
                use_container_width=True,
            )

        if perf.get("feature_importance"):
            st.subheader("Feature importance (selected model)")
            fi = pd.DataFrame(perf["feature_importance"])
            st.plotly_chart(
                charts.bar_chart(
                    fi, "feature", "importance", "Importance", horizontal=True, hover_fmt=".3f"
                ).update_layout(height=max(300, 24 * len(fi))),
                use_container_width=True,
            )
        if perf.get("split"):
            split = perf["split"]
            st.caption(
                f"Chronological split — train: {split['train_rows']:,} rows (before {split['validation_start']}), "
                f"validation: {split['validation_rows']:,} rows, test: {split['test_rows']:,} rows "
                f"(from {split['test_start']})."
            )
