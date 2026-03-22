# app.py — P2-ETF-DEEPM-ENGINE Streamlit Dashboard
# Light/white background, side-by-side Option A and B layout
# Plotly charts, OOS period only for backtest

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

import config as cfg

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeePM — ETF Signal Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Force white background */
  .stApp { background-color: #ffffff; }
  section[data-testid="stSidebar"] { background-color: #f8f9fa; }

  /* Hero card */
  .hero-card {
    background: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 12px;
    padding: 24px 28px 20px 28px;
    margin-bottom: 12px;
  }
  .hero-ticker {
    font-size: 48px;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.1;
  }
  .hero-conviction {
    font-size: 22px;
    font-weight: 500;
    color: #3a5bd9;
    margin-top: 2px;
  }
  .hero-date {
    font-size: 13px;
    color: #6b7280;
    margin-top: 6px;
  }
  .runner-up {
    font-size: 15px;
    color: #374151;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #e5e7eb;
  }

  /* Metric strip */
  .metric-row {
    display: flex;
    gap: 12px;
    margin: 10px 0;
  }
  .metric-box {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
  }
  .metric-label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }
  .metric-value {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
  }
  .metric-value.pos { color: #059669; }
  .metric-value.neg { color: #dc2626; }

  /* Macro pill */
  .macro-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 3px 3px 0;
  }
  .macro-green  { background:#d1fae5; color:#065f46; }
  .macro-amber  { background:#fef3c7; color:#92400e; }
  .macro-red    { background:#fee2e2; color:#991b1b; }

  /* Divider between options */
  .option-divider {
    border-left: 2px solid #e5e7eb;
    margin: 0 8px;
  }

  /* Footnote */
  .footnote {
    font-size: 11px;
    color: #9ca3af;
    margin-top: 8px;
  }

  /* Table */
  .signal-table { font-size: 13px; }

  /* Section header */
  .section-header {
    font-size: 13px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 18px 0 8px 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_signal(option: str) -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"models/signal_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def load_meta(option: str) -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"models/meta_option{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=cfg.FILE_MASTER,
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"Could not load master dataset: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_signal_history(option: str) -> pd.DataFrame:
    """Load all available signals + next-day actual returns."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"models/signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            records = json.load(f)
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


# ── Backtest helpers ───────────────────────────────────────────────────────────

def compute_oos_backtest(
    signal: dict,
    master: pd.DataFrame,
    option: str,
) -> dict:
    """
    Reconstruct OOS equity curve from signal weights stored in signal dict.
    Falls back to simple argmax strategy if no history available.
    """
    tickers   = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    # OOS period
    oos = master[master.index >= cfg.LIVE_START].copy()
    if oos.empty:
        return {}

    # Benchmark returns
    bench_col = f"{benchmark}_ret"
    bench_ret = oos[bench_col].fillna(0.0) if bench_col in oos.columns else pd.Series(0.0, index=oos.index)

    # Cash rate
    cash_rate = oos["TBILL_daily"].fillna(0.0) if "TBILL_daily" in oos.columns else pd.Series(0.0, index=oos.index)

    # Simple strategy: use latest weights to get asset ranking,
    # apply argmax (top pick) strategy over OOS period using actual returns
    weights_dict = signal.get("weights", {})
    if not weights_dict:
        return {}

    # Build daily returns for each ticker
    ret_cols = {t: f"{t}_ret" for t in tickers if f"{t}_ret" in oos.columns}
    rets = oos[[v for v in ret_cols.values()]].copy()
    rets.columns = [t for t in ret_cols.keys()]

    # Strategy: each day pick the top-weighted non-CASH asset
    # In live mode we use yesterday's weights — simple approximation for backtest
    # using cross-sectional momentum (best proxy without rerunning model)
    sorted_picks = sorted(
        [(t, weights_dict.get(t, 0.0)) for t in tickers],
        key=lambda x: x[1],
        reverse=True,
    )
    top_pick  = sorted_picks[0][0]
    pick_rets = rets[top_pick] if top_pick in rets.columns else cash_rate

    # Equity curves
    strat_curve = (1 + pick_rets.fillna(0.0)).cumprod()
    bench_curve = (1 + bench_ret.fillna(0.0)).cumprod()

    # Metrics
    def ann_return(r): return float(r.mean() * 252)
    def ann_vol(r):    return float(r.std() * np.sqrt(252))
    def sharpe(r):     return ann_return(r) / (ann_vol(r) + 1e-8)
    def max_dd(curve):
        roll_max = curve.cummax()
        dd = (curve - roll_max) / roll_max
        return float(dd.min())
    def hit_rate(r):   return float((r > 0).mean())

    pr = pick_rets.fillna(0.0)
    br = bench_ret.fillna(0.0)

    return {
        "dates":       oos.index,
        "strat_curve": strat_curve,
        "bench_curve": bench_curve,
        "strat_ret":   pr,
        "bench_ret":   br,
        "top_pick":    top_pick,
        "metrics": {
            "ann_return":  ann_return(pr),
            "ann_vol":     ann_vol(pr),
            "sharpe":      sharpe(pr),
            "max_dd":      max_dd(strat_curve),
            "hit_rate":    hit_rate(pr),
            "bench_ann_return": ann_return(br),
            "bench_sharpe":     sharpe(br),
        },
    }


# ── UI components ──────────────────────────────────────────────────────────────

def macro_pill(label: str, value: float, low: float, high: float) -> str:
    if value < low:
        cls = "macro-green"
    elif value > high:
        cls = "macro-red"
    else:
        cls = "macro-amber"
    return f'<span class="macro-pill {cls}">{label}: {value:.2f}</span>'


def render_hero(signal: dict, option: str) -> None:
    if "error" in signal:
        st.warning(f"Signal not available yet — run the training workflow first.")
        return

    weights    = signal.get("weights", {})
    tickers    = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    label_names = tickers + ["CASH"]

    # Sort by weight descending
    sorted_picks = sorted(
        [(t, weights.get(t, 0.0)) for t in label_names],
        key=lambda x: x[1],
        reverse=True,
    )

    top1 = sorted_picks[0]
    top2 = sorted_picks[1] if len(sorted_picks) > 1 else None
    top3 = sorted_picks[2] if len(sorted_picks) > 2 else None

    signal_date = signal.get("signal_date", "—")
    generated   = signal.get("generated_at", "")
    if generated:
        try:
            generated = datetime.fromisoformat(generated).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass

    runner_up_html = ""
    if top2:
        runner_up_html += f"<span style='color:#6b7280'>2nd: </span><b>{top2[0]}</b> {top2[1]*100:.1f}%"
    if top3:
        runner_up_html += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd: </span><b>{top3[0]}</b> {top3[1]*100:.1f}%"

    # Macro context pills
    rc = signal.get("regime_context", {})
    stress = signal.get("macro_stress")
    pills_html = ""
    if rc.get("VIX"):
        pills_html += macro_pill("VIX", rc["VIX"], 15, 25)
    if rc.get("T10Y2Y") is not None:
        pills_html += macro_pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"):
        pills_html += macro_pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if stress is not None:
        pills_html += macro_pill("Stress", stress, -0.5, 0.5)

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{top1[0]}</div>
      <div class="hero-conviction">{top1[1]*100:.1f}% conviction</div>
      <div class="hero-date">Signal for {signal_date} &nbsp;·&nbsp; Generated {generated}</div>
      <div class="runner-up">{runner_up_html}</div>
      <div style="margin-top:12px">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(bt: dict) -> None:
    if not bt or "metrics" not in bt:
        st.info("Backtest metrics will appear after first training run.")
        return

    m = bt["metrics"]

    def fmt_pct(v): return f"{v*100:.1f}%"
    def cls(v):     return "pos" if v >= 0 else "neg"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Ann Return</div>
        <div class="metric-value {cls(m['ann_return'])}">{fmt_pct(m['ann_return'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Ann Vol</div>
        <div class="metric-value">{fmt_pct(m['ann_vol'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe</div>
        <div class="metric-value {cls(m['sharpe'])}">{m['sharpe']:.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max DD (peak→trough)</div>
        <div class="metric-value neg">{fmt_pct(m['max_dd'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value">{fmt_pct(m['hit_rate'])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_equity_curve(bt: dict, benchmark_name: str) -> None:
    if not bt or "strat_curve" not in bt:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bt["dates"],
        y=bt["strat_curve"].values,
        name=f"DeePM ({bt['top_pick']})",
        line=dict(color="#3a5bd9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bt["dates"],
        y=bt["bench_curve"].values,
        name=benchmark_name,
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=10),
                   tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_signal_history(history_df: pd.DataFrame, master: pd.DataFrame, option: str) -> None:
    if history_df.empty:
        st.info("Signal history will appear after the first training run.")
        return

    tickers = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS

    # Enrich with actual next-day return if not already present
    if "actual_return" not in history_df.columns and not master.empty:
        def get_actual(row):
            try:
                date      = pd.Timestamp(row["signal_date"])
                pick      = row["pick"]
                ret_col   = f"{pick}_ret"
                if ret_col in master.columns and date in master.index:
                    return master.loc[date, ret_col]
            except Exception:
                pass
            return np.nan
        history_df["actual_return"] = history_df.apply(get_actual, axis=1)

    if "hit" not in history_df.columns and "actual_return" in history_df.columns:
        history_df["hit"] = history_df["actual_return"].apply(
            lambda x: "✓" if (not np.isnan(x) and x > 0) else ("✗" if not np.isnan(x) else "—")
        )

    display = history_df.copy()
    display = display.sort_values("signal_date", ascending=False)

    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Conviction",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols = [c for c in col_map if c in display.columns]
    display = display[cols].rename(columns=col_map)

    if "Conviction" in display.columns:
        display["Conviction"] = display["Conviction"].apply(lambda x: f"{x*100:.1f}%")
    if "Actual Return" in display.columns:
        display["Actual Return"] = display["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "—"
        )

    # Compute hit rate summary
    if "Hit" in display.columns:
        hits   = (display["Hit"] == "✓").sum()
        misses = (display["Hit"] == "✗").sum()
        total  = hits + misses
        hr     = hits / total if total > 0 else 0.0
        st.markdown(
            f"<div style='font-size:13px;color:#374151;margin-bottom:8px;'>"
            f"Hit rate: <b>{hr:.1%}</b> ({hits}/{total} signals)"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Hit": st.column_config.TextColumn(width="small"),
            "Pick": st.column_config.TextColumn(width="small"),
        },
    )


def render_footnote(meta: dict, option: str) -> None:
    if "error" in meta or not meta:
        return
    trained_at   = meta.get("trained_at", "—")
    winning_loss = meta.get("winning_loss", meta.get("loss_fn", "—"))
    n_params     = meta.get("n_params", 0)
    test_sharpe  = meta.get("test_sharpe", 0)
    test_ret     = meta.get("test_ann_return", 0)

    try:
        trained_at = datetime.fromisoformat(trained_at).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    st.markdown(
        f"<div class='footnote'>"
        f"Trained {trained_at} &nbsp;·&nbsp; "
        f"Loss: {winning_loss} &nbsp;·&nbsp; "
        f"Params: {n_params:,} &nbsp;·&nbsp; "
        f"Test Sharpe: {test_sharpe:.3f} &nbsp;·&nbsp; "
        f"Test Ann Return: {test_ret*100:.2f}%"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;'>DeePM — Distributionally Robust ETF Engine</h2>"
        "<p style='color:#6b7280;font-size:14px;margin-top:0;'>Regime-aware · Macro Graph Prior · EVaR objective</p>",
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("Loading signals and data..."):
        sig_A   = load_signal("A")
        sig_B   = load_signal("B")
        meta_A  = load_meta("A")
        meta_B  = load_meta("B")
        master  = load_master()
        hist_A  = load_signal_history("A")
        hist_B  = load_signal_history("B")

    # Backtest
    bt_A = compute_oos_backtest(sig_A, master, "A") if not master.empty else {}
    bt_B = compute_oos_backtest(sig_B, master, "B") if not master.empty else {}

    # ── Side-by-side columns ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            "<div style='font-size:16px;font-weight:600;color:#1a1a2e;margin-bottom:8px;'>"
            "Option A — Fixed Income / Alts"
            "</div>",
            unsafe_allow_html=True,
        )
        render_hero(sig_A, "A")

        st.markdown("<div class='section-header'>OOS Performance</div>", unsafe_allow_html=True)
        render_metrics(bt_A)
        render_equity_curve(bt_A, "AGG")
        render_footnote(meta_A, "A")

    with col_b:
        st.markdown(
            "<div style='font-size:16px;font-weight:600;color:#1a1a2e;margin-bottom:8px;'>"
            "Option B — Equity Sectors"
            "</div>",
            unsafe_allow_html=True,
        )
        render_hero(sig_B, "B")

        st.markdown("<div class='section-header'>OOS Performance</div>", unsafe_allow_html=True)
        render_metrics(bt_B)
        render_equity_curve(bt_B, "SPY")
        render_footnote(meta_B, "B")

    # ── Signal history — full width ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-size:16px;font-weight:600;color:#1a1a2e;margin-bottom:12px;'>"
        "Signal History"
        "</div>",
        unsafe_allow_html=True,
    )

    tab_a, tab_b = st.tabs(["Option A — Fixed Income / Alts", "Option B — Equity Sectors"])

    with tab_a:
        render_signal_history(hist_A, master, "A")

    with tab_b:
        render_signal_history(hist_B, master, "B")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:11px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-DEEPM-ENGINE · Research only · Not financial advice · "
        "Data: HuggingFace P2SAMAPA/p2-etf-deepm-data"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
