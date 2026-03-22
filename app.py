# app.py — P2-ETF-DEEPM-ENGINE Streamlit Dashboard
# Tab layout: Option A | Option B
# Hero shows BEST signal across fixed split and shrinking window
# Backtest correctly handles CASH as a valid pick

import json
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

  .hero-card {
    background: #f0f4ff; border: 1px solid #d0d8f0;
    border-radius: 14px; padding: 28px 32px 22px 32px; margin-bottom: 24px;
  }
  .hero-ticker { font-size: 64px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
  .hero-conv   { font-size: 28px; font-weight: 500; color: #3a5bd9; margin-top: 6px; }
  .hero-date   { font-size: 15px; color: #6b7280; margin-top: 8px; }
  .hero-source { font-size: 14px; color: #3a5bd9; font-weight: 600;
                 background: #e0e7ff; border-radius: 20px; padding: 3px 12px;
                 display: inline-block; margin-top: 8px; }
  .runner-up   { font-size: 18px; color: #374151; margin-top: 14px;
                 padding-top: 14px; border-top: 1px solid #e5e7eb; }

  .label-fixed  { display:inline-block; font-size:14px; font-weight:700;
                  color:#374151; text-transform:uppercase; letter-spacing:.07em;
                  background:#f3f4f6; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .label-window { display:inline-block; font-size:14px; font-weight:700;
                  color:#1d4ed8; text-transform:uppercase; letter-spacing:.07em;
                  background:#eff6ff; border-radius:6px;
                  padding:5px 14px; margin-bottom:12px; }
  .window-badge { font-size:14px; color:#1d4ed8; background:#eff6ff;
                  border:1px solid #bfdbfe; border-radius:20px;
                  padding:4px 14px; display:inline-block; margin-bottom:12px; }

  .metric-row { display:flex; gap:12px; margin:10px 0 16px 0; }
  .metric-box { flex:1; background:#fff; border:1px solid #e5e7eb;
                border-radius:10px; padding:14px 10px; text-align:center; }
  .metric-label { font-size:12px; color:#6b7280; text-transform:uppercase;
                  letter-spacing:.05em; margin-bottom:6px; }
  .metric-value { font-size:24px; font-weight:600; color:#111827; }
  .pos { color:#059669; } .neg { color:#dc2626; }

  .pill   { display:inline-block; padding:5px 14px; border-radius:20px;
            font-size:15px; font-weight:500; margin:3px 3px 3px 0; }
  .pill-g { background:#d1fae5; color:#065f46; }
  .pill-a { background:#fef3c7; color:#92400e; }
  .pill-r { background:#fee2e2; color:#991b1b; }

  .hit-line { font-size:16px; color:#374151; margin-bottom:10px; }
  .fn       { font-size:13px; color:#9ca3af; margin-top:8px; }
  .sec-hdr  { font-size:20px; font-weight:700; color:#1a1a2e; margin:28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_json(filename: str) -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO, filename=filename,
            repo_type="dataset", token=cfg.HF_TOKEN or None, force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO, filename=cfg.FILE_MASTER,
            repo_type="dataset", token=cfg.HF_TOKEN or None, force_download=True,
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
def load_history(option: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=f"models/signal_history_{option}.json",
            repo_type="dataset", token=cfg.HF_TOKEN or None, force_download=True,
        )
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    except Exception:
        return pd.DataFrame()


def load_signals() -> dict:
    raw = _load_json("models/latest_signals.json")
    return {
        "A":  raw.get("option_A")  or {},
        "B":  raw.get("option_B")  or {},
        "Aw": raw.get("option_A_window") or {},
        "Bw": raw.get("option_B_window") or {},
    }


# ── Backtest ───────────────────────────────────────────────────────────────────

def build_bt(signal: dict, master: pd.DataFrame, option: str) -> dict:
    """
    Build OOS equity curve from signal.
    Respects CASH as a valid pick — uses T-bill rate if pick is CASH.
    """
    if not signal or "weights" not in signal or master.empty:
        return {}

    tickers   = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK
    label_names = tickers + ["CASH"]

    oos = master[master.index >= cfg.LIVE_START].copy()
    if oos.empty:
        return {}

    bench_ret = oos.get(f"{benchmark}_ret",
                        pd.Series(0.0, index=oos.index)).fillna(0.0)
    cash_rate = oos.get("TBILL_daily",
                        pd.Series(0.0, index=oos.index)).fillna(0.0)

    # Find true top pick including CASH
    weights_dict = signal.get("weights", {})
    all_picks = sorted(
        [(t, weights_dict.get(t, 0.0)) for t in label_names],
        key=lambda x: x[1], reverse=True,
    )
    top_pick = all_picks[0][0]

    # Get returns for top pick
    if top_pick == "CASH":
        pick_rets  = cash_rate
        pick_label = "CASH (T-bill)"
    else:
        ret_col   = f"{top_pick}_ret"
        pick_rets = oos.get(ret_col, cash_rate).fillna(0.0)
        pick_label = top_pick

    sc = (1 + pick_rets).cumprod()
    bc = (1 + bench_ret).cumprod()

    def ar(r):  return float(r.mean() * 252)
    def av(r):  return float(r.std() * np.sqrt(252))
    def sh(r):  return ar(r) / (av(r) + 1e-8)
    def dd(c):  return float(((c - c.cummax()) / c.cummax()).min())
    def hr(r):  return float((r > 0).mean())

    return {
        "dates":       oos.index,
        "sc":          sc,
        "bc":          bc,
        "top_pick":    top_pick,
        "pick_label":  pick_label,
        "benchmark":   benchmark,
        "m": {
            "ar": ar(pick_rets),
            "av": av(pick_rets),
            "sh": sh(pick_rets),
            "dd": dd(sc),
            "hr": hr(pick_rets),
        },
    }


def best_signal(sig_fixed: dict, sig_window: dict) -> tuple:
    """
    Return (best_signal, source_label) where best = higher OOS ann return.
    Fixed split uses test_ann_return; shrinking window uses oos_ann_return.
    """
    ret_fixed  = sig_fixed.get("test_ann_return", -999) if sig_fixed else -999
    ret_window = sig_window.get("oos_ann_return",  -999) if sig_window else -999

    if ret_window > ret_fixed and sig_window and "pick" in sig_window:
        return sig_window, "Shrinking Window"
    elif sig_fixed and "pick" in sig_fixed:
        return sig_fixed, "Fixed Split"
    else:
        return sig_fixed or sig_window or {}, "—"


# ── UI helpers ─────────────────────────────────────────────────────────────────

def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="pill {cls}">{label}: {val}</span>'


def render_hero(sig_fixed: dict, sig_window: dict, option: str):
    best, source = best_signal(sig_fixed, sig_window)

    if not best or "pick" not in best:
        st.info("Signal not available yet — run the training workflow first.")
        return

    tickers     = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    label_names = tickers + ["CASH"]
    w           = best.get("weights", {})
    picks       = sorted(
        [(t, w.get(t, 0.0)) for t in label_names],
        key=lambda x: x[1], reverse=True,
    )

    t1 = picks[0]
    t2 = picks[1] if len(picks) > 1 else None
    t3 = picks[2] if len(picks) > 2 else None

    sig_date = best.get("signal_date", "—")
    gen      = best.get("generated_at", "")
    try:
        gen = datetime.fromisoformat(gen).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    runner = ""
    if t2: runner += f"<span style='color:#6b7280'>2nd:</span> <b>{t2[0]}</b> {t2[1]*100:.1f}%"
    if t3: runner += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd:</span> <b>{t3[0]}</b> {t3[1]*100:.1f}%"

    rc  = best.get("regime_context", {})
    st_ = best.get("macro_stress")
    pills = ""
    if rc.get("VIX"):              pills += pill("VIX",    rc["VIX"],    15,   25)
    if rc.get("T10Y2Y") is not None: pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"):        pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if st_ is not None:            pills += pill("Stress",  st_,          -0.5, 0.5)

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-ticker">{t1[0]}</div>
      <div class="hero-conv">{t1[1]*100:.1f}% conviction</div>
      <div class="hero-date">Signal for {sig_date} &nbsp;·&nbsp; Generated {gen}</div>
      <div class="hero-source">Source: {source}</div>
      <div class="runner-up">{runner}</div>
      <div style="margin-top:16px">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(bt: dict):
    if not bt:
        st.caption("Metrics available after first training run.")
        return
    m  = bt["m"]
    fp = lambda v: f"{v*100:.1f}%"
    c  = lambda v: "pos" if v >= 0 else "neg"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Ann Return</div>
        <div class="metric-value {c(m['ar'])}">{fp(m['ar'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Ann Vol</div>
        <div class="metric-value">{fp(m['av'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe</div>
        <div class="metric-value {c(m['sh'])}">{m['sh']:.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max DD (peak→trough)</div>
        <div class="metric-value neg">{fp(m['dd'])}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Hit Rate</div>
        <div class="metric-value">{fp(m['hr'])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_curve(bt: dict):
    if not bt:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["sc"].values,
        name=f"DeePM ({bt['pick_label']})",
        line=dict(color="#3a5bd9", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["bc"].values,
        name=bt["benchmark"],
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickfont=dict(size=12),
                   tickformat=".2f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_footnote(signal: dict, window: bool = False):
    if not signal:
        return
    trained = signal.get("trained_at", "—")
    try:
        trained = datetime.fromisoformat(trained).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    if window:
        wid  = signal.get("winning_window", "?")
        ws   = signal.get("winning_train_start", "?")
        we   = signal.get("winning_train_end", "?")
        loss = signal.get("winning_loss", "—")
        ret  = signal.get("oos_ann_return", 0)
        shr  = signal.get("oos_sharpe", 0)
        detail = (f"Window {wid} ({ws}→{we}) &nbsp;·&nbsp; Loss: {loss} &nbsp;·&nbsp; "
                  f"OOS Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}")
    else:
        loss = signal.get("winning_loss", "—")
        ret  = signal.get("test_ann_return", 0)
        shr  = signal.get("test_sharpe", 0)
        detail = (f"Loss: {loss} &nbsp;·&nbsp; "
                  f"Test Return: {ret*100:.2f}% &nbsp;·&nbsp; Sharpe: {shr:.3f}")

    st.markdown(
        f"<div class='fn'>Trained {trained} &nbsp;·&nbsp; {detail}</div>",
        unsafe_allow_html=True,
    )


def render_history(hist_df: pd.DataFrame, master: pd.DataFrame):
    if hist_df.empty:
        st.info("Signal history will appear after the first training run.")
        return

    if "actual_return" not in hist_df.columns and not master.empty:
        def get_ret(row):
            try:
                date = pd.Timestamp(row["signal_date"])
                pick = row["pick"]
                if pick == "CASH":
                    col = "TBILL_daily"
                else:
                    col = f"{pick}_ret"
                if col in master.columns and date in master.index:
                    return master.loc[date, col]
            except Exception:
                pass
            return np.nan
        hist_df["actual_return"] = hist_df.apply(get_ret, axis=1)

    if "hit" not in hist_df.columns and "actual_return" in hist_df.columns:
        hist_df["hit"] = hist_df["actual_return"].apply(
            lambda x: "✓" if (not np.isnan(x) and x > 0) else ("✗" if not np.isnan(x) else "—")
        )

    disp = hist_df.sort_values("signal_date", ascending=False).copy()
    col_map = {
        "signal_date":   "Date",
        "pick":          "Pick",
        "conviction":    "Conviction",
        "actual_return": "Actual Return",
        "hit":           "Hit",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Conviction" in disp.columns:
        disp["Conviction"] = disp["Conviction"].apply(lambda x: f"{x*100:.1f}%")
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if not np.isnan(x) else "—"
        )

    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div class='hit-line'>Hit rate: <b>{hr:.1%}</b> &nbsp;({hits}/{total} signals)</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={
                     "Hit":  st.column_config.TextColumn(width="small"),
                     "Pick": st.column_config.TextColumn(width="small"),
                 })


# ── Option renderer ────────────────────────────────────────────────────────────

def render_option(option: str, signals: dict, master: pd.DataFrame):
    sig  = signals.get(option,         {})
    sigw = signals.get(f"{option}w",   {})
    hist = load_history(option)

    # Hero — best of fixed split vs shrinking window
    render_hero(sig, sigw, option)

    # Side by side performance panels
    col_f, col_w = st.columns(2, gap="large")

    bt_f = build_bt(sig,  master, option)
    bt_w = build_bt(sigw, master, option)

    with col_f:
        st.markdown("<div class='label-fixed'>Fixed Split (70/15/15)</div>",
                    unsafe_allow_html=True)
        render_metrics(bt_f)
        render_curve(bt_f)
        render_footnote(sig, window=False)

    with col_w:
        st.markdown("<div class='label-window'>Shrinking Window</div>",
                    unsafe_allow_html=True)
        if sigw and "winning_window" in sigw:
            st.markdown(
                f"<div class='window-badge'>"
                f"Window {sigw['winning_window']}: "
                f"{sigw.get('winning_train_start','?')} → {sigw.get('winning_train_end','?')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        render_metrics(bt_w)
        render_curve(bt_w)
        render_footnote(sigw, window=True)

    # Signal history — full width
    st.markdown("<div class='sec-hdr'>Signal History</div>", unsafe_allow_html=True)
    render_history(hist, master)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;font-size:34px;'>"
        "DeePM — Distributionally Robust ETF Engine</h2>"
        "<p style='color:#6b7280;font-size:16px;margin-top:0;'>"
        "Regime-aware &nbsp;·&nbsp; Macro Graph Prior &nbsp;·&nbsp; EVaR objective</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading signals and data..."):
        signals = load_signals()
        master  = load_master()

    tab_a, tab_b = st.tabs([
        "📊  Option A — Fixed Income / Alts",
        "📈  Option B — Equity Sectors",
    ])

    with tab_a:
        render_option("A", signals, master)

    with tab_b:
        render_option("B", signals, master)

    st.markdown(
        "<div style='margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:13px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-DEEPM-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
