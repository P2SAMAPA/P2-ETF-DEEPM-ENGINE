# app.py — P2-ETF-DEEPM-ENGINE Streamlit Dashboard
# Light/white background
# Layout: Option A (left) | Option B (right)
# Each option: hero card + fixed split vs shrinking window side by side

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

import config as cfg

st.set_page_config(
    page_title="DeePM — ETF Signal Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stApp { background-color: #ffffff; }
  section[data-testid="stSidebar"] { background-color: #f8f9fa; }

  /* Hero card */
  .hero-card {
    background: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 12px;
    padding: 24px 28px 20px 28px;
    margin-bottom: 16px;
  }
  .hero-ticker  { font-size: 56px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
  .hero-conv    { font-size: 26px; font-weight: 500; color: #3a5bd9; margin-top: 4px; }
  .hero-date    { font-size: 15px; color: #6b7280; margin-top: 8px; }
  .runner-up    { font-size: 17px; color: #374151; margin-top: 12px;
                  padding-top: 12px; border-top: 1px solid #e5e7eb; }

  /* Section label */
  .split-label {
    font-size: 13px; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 4px 0 10px 0; padding: 4px 10px;
    background: #f3f4f6; border-radius: 6px; display: inline-block;
  }
  .window-label {
    font-size: 13px; font-weight: 700; color: #1d4ed8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 4px 0 10px 0; padding: 4px 10px;
    background: #eff6ff; border-radius: 6px; display: inline-block;
  }
  .window-badge {
    font-size: 13px; color: #1d4ed8;
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 20px; padding: 3px 12px;
    display: inline-block; margin-bottom: 10px;
  }

  /* Metric strip */
  .metric-row { display: flex; gap: 10px; margin: 10px 0; }
  .metric-box {
    flex: 1; background: #ffffff; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 12px 10px; text-align: center;
  }
  .metric-label { font-size: 12px; color: #6b7280; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 600; color: #111827; }
  .metric-value.pos { color: #059669; }
  .metric-value.neg { color: #dc2626; }

  /* Macro pills */
  .macro-pill { display: inline-block; padding: 4px 12px; border-radius: 20px;
                font-size: 14px; font-weight: 500; margin: 3px 3px 3px 0; }
  .macro-green { background: #d1fae5; color: #065f46; }
  .macro-amber { background: #fef3c7; color: #92400e; }
  .macro-red   { background: #fee2e2; color: #991b1b; }

  /* Section header */
  .section-hdr {
    font-size: 14px; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 18px 0 8px 0;
  }

  /* Option header */
  .option-hdr {
    font-size: 20px; font-weight: 700; color: #1a1a2e; margin-bottom: 10px;
  }

  /* Footnote */
  .footnote { font-size: 13px; color: #9ca3af; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _hf_load_json(filename: str) -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=filename,
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


def load_all_signals() -> dict:
    raw = _hf_load_json("models/latest_signals.json")
    return {
        "A":        raw.get("option_A", {}),
        "B":        raw.get("option_B", {}),
        "A_window": raw.get("option_A_window", {}),
        "B_window": raw.get("option_B_window", {}),
    }


def load_meta(option: str, window: bool = False) -> dict:
    suffix = "_window" if window else ""
    return _hf_load_json(f"models/meta_option{option}{suffix}.json")


def load_signal_history(option: str) -> pd.DataFrame:
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

def oos_equity_curve(signal: dict, master: pd.DataFrame, option: str) -> dict:
    """Build OOS equity curve from signal weights."""
    if not signal or "weights" not in signal or master.empty:
        return {}

    tickers   = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    oos = master[master.index >= cfg.LIVE_START].copy()
    if oos.empty:
        return {}

    bench_ret = oos.get(f"{benchmark}_ret", pd.Series(0.0, index=oos.index)).fillna(0.0)
    cash_rate = oos.get("TBILL_daily", pd.Series(0.0, index=oos.index)).fillna(0.0)

    weights_dict = signal.get("weights", {})
    sorted_picks = sorted(
        [(t, weights_dict.get(t, 0.0)) for t in tickers],
        key=lambda x: x[1], reverse=True,
    )
    top_pick  = sorted_picks[0][0]
    ret_col   = f"{top_pick}_ret"
    pick_rets = oos[ret_col].fillna(0.0) if ret_col in oos.columns else cash_rate

    strat_curve = (1 + pick_rets).cumprod()
    bench_curve = (1 + bench_ret).cumprod()

    def ann_ret(r):  return float(r.mean() * 252)
    def ann_vol(r):  return float(r.std() * np.sqrt(252))
    def sharpe(r):   return ann_ret(r) / (ann_vol(r) + 1e-8)
    def max_dd(c):
        roll_max = c.cummax()
        return float(((c - roll_max) / roll_max).min())
    def hit_rate(r): return float((r > 0).mean())

    return {
        "dates":       oos.index,
        "strat_curve": strat_curve,
        "bench_curve": bench_curve,
        "top_pick":    top_pick,
        "benchmark":   benchmark,
        "metrics": {
            "ann_return": ann_ret(pick_rets),
            "ann_vol":    ann_vol(pick_rets),
            "sharpe":     sharpe(pick_rets),
            "max_dd":     max_dd(strat_curve),
            "hit_rate":   hit_rate(pick_rets),
        },
    }


# ── UI components ──────────────────────────────────────────────────────────────

def macro_pill_html(label, value, low, high):
    cls = "macro-green" if value < low else ("macro-red" if value > high else "macro-amber")
    return f'<span class="macro-pill {cls}">{label}: {value}</span>'


def render_hero(signal: dict, option: str):
    if not signal or "error" in signal or "pick" not in signal:
        st.info("Signal not available yet — run the training workflow first.")
        return

    weights     = signal.get("weights", {})
    tickers     = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    label_names = tickers + ["CASH"]

    sorted_picks = sorted(
        [(t, weights.get(t, 0.0)) for t in label_names],
        key=lambda x: x[1], reverse=True,
    )
    top1 = sorted_picks[0]
    top2 = sorted_picks[1] if len(sorted_picks) > 1 else None
    top3 = sorted_picks[2] if len(sorted_picks) > 2 else None

    signal_date = signal.get("signal_date", "—")
    generated   = signal.get("generated_at", "")
    try:
        generated = datetime.fromisoformat(generated).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    runner_html = ""
    if top2:
        runner_html += f"<span style='color:#6b7280'>2nd:</span> <b>{top2[0]}</b> {top2[1]*100:.1f}%"
    if top3:
        runner_html += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd:</span> <b>{top3[0]}</b> {top3[1]*100:.1f}%"

    rc = signal.get("regime_context", {})
    stress = signal.get("macro_stress")
    pills  = ""
    if rc.get("VIX"):        pills += macro_pill_html("VIX", rc["VIX"], 15, 25)
    if rc.get("T10Y2Y") is not None: pills += macro_pill_html("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"):  pills += macro_pill_html("HY Spr", rc["HY_SPREAD"], 300, 500)
    if stress is not None:   pills += macro_pill_html("Stress", stress, -0.5, 0.5)

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{top1[0]}</div>
      <div class="hero-conv">{top1[1]*100:.1f}% conviction</div>
      <div class="hero-date">Signal for {signal_date} &nbsp;·&nbsp; Generated {generated}</div>
      <div class="runner-up">{runner_html}</div>
      <div style="margin-top:14px">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(bt: dict):
    if not bt or "metrics" not in bt:
        st.caption("Metrics available after first training run.")
        return
    m = bt["metrics"]
    def fp(v): return f"{v*100:.1f}%"
    def cls(v): return "pos" if v >= 0 else "neg"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Ann Return</div>
        <div class="metric-value {cls(m['ann_return'])}">{fp(m['ann_return'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Ann Vol</div>
        <div class="metric-value">{fp(m['ann_vol'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe</div>
        <div class="metric-value {cls(m['sharpe'])}">{m['sharpe']:.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max DD (peak→trough)</div>
        <div class="metric-value neg">{fp(m['max_dd'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value">{fp(m['hit_rate'])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_curve(bt: dict):
    if not bt or "strat_curve" not in bt:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["strat_curve"].values,
        name=f"DeePM ({bt['top_pick']})",
        line=dict(color="#3a5bd9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["bench_curve"].values,
        name=bt["benchmark"],
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=12)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=11),
                   tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_footnote(signal: dict, meta: dict, window: bool = False):
    if not signal or not meta or "error" in meta:
        return
    trained_at = signal.get("trained_at", "—")
    try:
        trained_at = datetime.fromisoformat(trained_at).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    if window:
        wstart = signal.get("winning_train_start", "?")
        wend   = signal.get("winning_train_end", "?")
        wid    = signal.get("winning_window", "?")
        loss   = signal.get("winning_loss", "—")
        ret    = signal.get("oos_ann_return", 0)
        shr    = signal.get("oos_sharpe", 0)
        extra  = (f"Window {wid} ({wstart}→{wend}) &nbsp;·&nbsp; "
                  f"Loss: {loss} &nbsp;·&nbsp; "
                  f"OOS Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}")
    else:
        loss = signal.get("winning_loss", "—")
        ret  = signal.get("test_ann_return", 0)
        shr  = signal.get("test_sharpe", 0)
        extra = (f"Loss: {loss} &nbsp;·&nbsp; "
                 f"Test Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}")

    st.markdown(
        f"<div class='footnote'>Trained {trained_at} &nbsp;·&nbsp; {extra}</div>",
        unsafe_allow_html=True,
    )


def render_signal_history(history_df: pd.DataFrame, master: pd.DataFrame, option: str):
    if history_df.empty:
        st.info("Signal history will appear after the first training run.")
        return

    if "actual_return" not in history_df.columns and not master.empty:
        def get_actual(row):
            try:
                date    = pd.Timestamp(row["signal_date"])
                ret_col = f"{row['pick']}_ret"
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

    display = history_df.sort_values("signal_date", ascending=False).copy()

    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Conviction",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols    = [c for c in col_map if c in display.columns]
    display = display[cols].rename(columns=col_map)

    if "Conviction" in display.columns:
        display["Conviction"] = display["Conviction"].apply(lambda x: f"{x*100:.1f}%")
    if "Actual Return" in display.columns:
        display["Actual Return"] = display["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "—"
        )

    if "Hit" in display.columns:
        hits   = (display["Hit"] == "✓").sum()
        total  = (display["Hit"].isin(["✓", "✗"])).sum()
        hr     = hits / total if total > 0 else 0.0
        st.markdown(
            f"<div style='font-size:15px;color:#374151;margin-bottom:8px;'>"
            f"Hit rate: <b>{hr:.1%}</b> &nbsp;({hits}/{total} signals)"
            f"</div>", unsafe_allow_html=True,
        )

    st.dataframe(display, use_container_width=True, hide_index=True,
                 column_config={
                     "Hit":  st.column_config.TextColumn(width="small"),
                     "Pick": st.column_config.TextColumn(width="small"),
                 })


# ── Main app ───────────────────────────────────────────────────────────────────

def render_option(option: str, signals: dict, master: pd.DataFrame):
    """Render one full option column (hero + fixed split | shrinking window + history)."""
    label = "Option A — Fixed Income / Alts" if option == "A" else "Option B — Equity Sectors"
    st.markdown(f"<div class='option-hdr'>{label}</div>", unsafe_allow_html=True)

    sig        = signals.get(option, {})
    sig_window = signals.get(f"{option}_window", {})
    meta       = load_meta(option)
    meta_win   = load_meta(option, window=True)

    # Hero card — use fixed split signal (primary)
    render_hero(sig, option)

    # Side by side: fixed split | shrinking window
    col_fixed, col_win = st.columns(2, gap="medium")

    bt_fixed  = oos_equity_curve(sig,        master, option)
    bt_window = oos_equity_curve(sig_window, master, option)

    with col_fixed:
        st.markdown("<div class='split-label'>Fixed Split (70/15/15)</div>", unsafe_allow_html=True)
        render_metrics(bt_fixed)
        render_curve(bt_fixed)
        render_footnote(sig, meta, window=False)

    with col_win:
        st.markdown("<div class='window-label'>Shrinking Window</div>", unsafe_allow_html=True)
        if sig_window and "winning_window" in sig_window:
            st.markdown(
                f"<div class='window-badge'>"
                f"Window {sig_window['winning_window']}: "
                f"{sig_window.get('winning_train_start','?')} → {sig_window.get('winning_train_end','?')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        render_metrics(bt_window)
        render_curve(bt_window)
        render_footnote(sig_window, meta_win, window=True)


def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;font-size:32px;'>"
        "DeePM — Distributionally Robust ETF Engine</h2>"
        "<p style='color:#6b7280;font-size:16px;margin-top:0;'>"
        "Regime-aware &nbsp;·&nbsp; Macro Graph Prior &nbsp;·&nbsp; EVaR objective</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading signals and data..."):
        signals = load_all_signals()
        master  = load_master()
        hist_A  = load_signal_history("A")
        hist_B  = load_signal_history("B")

    # Side by side options
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        render_option("A", signals, master)

    with col_b:
        render_option("B", signals, master)

    # Signal history — full width
    st.markdown("---")
    st.markdown(
        "<div style='font-size:20px;font-weight:700;color:#1a1a2e;margin-bottom:14px;'>"
        "Signal History</div>",
        unsafe_allow_html=True,
    )

    tab_a, tab_b = st.tabs([
        "Option A — Fixed Income / Alts",
        "Option B — Equity Sectors",
    ])
    with tab_a:
        render_signal_history(hist_A, master, "A")
    with tab_b:
        render_signal_history(hist_B, master, "B")

    st.markdown(
        "<div style='margin-top:32px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:13px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-DEEPM-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
