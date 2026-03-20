"""
ob_detector.py
──────────────────────────────────────────────────────────────
ICT Order Block Detector

ICT Concept:
  An Order Block (OB) is the last opposing candle before a significant
  impulsive move. It represents an area where institutional orders were
  placed. Price often returns to these zones for a reaction.

  Bullish OB  → last BEARISH (red) candle before a strong bullish impulse
  Bearish OB  → last BULLISH (green) candle before a strong bearish impulse

  A "significant impulse" is defined here as a move that:
    1. Breaks the prior swing high/low (BOS confirmation), OR
    2. Moves at least N * ATR in one direction within M candles

  Status tracking:
    - Fresh     : OB has not been tested (price hasn't returned to zone)
    - Tested    : Price has entered the OB zone at least once
    - Mitigated : Price has closed beyond the OB zone (OB consumed)

Usage:
  python ob_detector.py                          # default: multi-ticker
  python ob_detector.py --ticker AAPL            # single ticker
  python ob_detector.py --ticker NQ=F --tf 1h   # specific timeframe
  python ob_detector.py --help
"""

import argparse
import warnings
import sys

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["NQ=F", "EURUSD=X", "AAPL", "SPY"]

TIMEFRAME_MAP = {
    "1h":  {"period": "180d",  "label": "1H"},
    "4h":  {"period": "365d",  "label": "4H"},
    "1d":  {"period": "5y",    "label": "Daily"},
}

# Impulse strength: minimum ATR multiplier to qualify as an impulse
IMPULSE_ATR_MULT  = 1.8

# How many candles forward to look for the impulse after an OB candle
IMPULSE_LOOKFORWARD = 3

# Minimum ATR multiple for the OB zone height (filters micro OBs)
MIN_OB_SIZE_ATR   = 0.2

# Max number of OBs to keep per side (most recent wins)
MAX_OBS_PER_SIDE  = 30

# Colour scheme
BULL_FILL   = "#34d399"
BULL_EDGE   = "#059669"
BEAR_FILL   = "#fb7185"
BEAR_EDGE   = "#be123c"
TESTED_ALPHA    = 0.25
FRESH_ALPHA     = 0.18
MITIGATED_ALPHA = 0.08
BG          = "#0d1117"
PANEL       = "#161b22"
GRID_C      = "#21262d"
TEXT_C      = "#c9d1d9"
MUTED_C     = "#8b949e"
UP_CANDLE   = "#34d399"
DOWN_CANDLE = "#fb7185"


# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("4h").agg({
        "Open": "first", "High": "max",
        "Low": "min", "Close": "last",
        "Volume": "sum"
    }).dropna()


# ─────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def is_bullish(row) -> bool:
    return float(row["Close"]) > float(row["Open"])


def is_bearish(row) -> bool:
    return float(row["Close"]) < float(row["Open"])


def candle_body_size(row) -> float:
    return abs(float(row["Close"]) - float(row["Open"]))


# ─────────────────────────────────────────────────────────
# IMPULSE DETECTION
# ─────────────────────────────────────────────────────────

def detect_bullish_impulse(df: pd.DataFrame, start_idx: int,
                            atr: pd.Series) -> bool:
    """
    Returns True if there is a bullish impulse starting at start_idx.
    An impulse = cumulative upward move > IMPULSE_ATR_MULT * ATR
    within IMPULSE_LOOKFORWARD candles.
    """
    if start_idx >= len(df):
        return False
    atr_val = float(atr.iloc[start_idx])
    base    = float(df["Low"].iloc[start_idx])
    end     = min(start_idx + IMPULSE_LOOKFORWARD + 1, len(df))
    highs   = df["High"].iloc[start_idx + 1 : end].values
    if len(highs) == 0:
        return False
    peak   = float(np.max(highs))
    return (peak - base) >= IMPULSE_ATR_MULT * atr_val


def detect_bearish_impulse(df: pd.DataFrame, start_idx: int,
                             atr: pd.Series) -> bool:
    """
    Returns True if there is a bearish impulse starting at start_idx.
    """
    if start_idx >= len(df):
        return False
    atr_val = float(atr.iloc[start_idx])
    base    = float(df["High"].iloc[start_idx])
    end     = min(start_idx + IMPULSE_LOOKFORWARD + 1, len(df))
    lows    = df["Low"].iloc[start_idx + 1 : end].values
    if len(lows) == 0:
        return False
    trough = float(np.min(lows))
    return (base - trough) >= IMPULSE_ATR_MULT * atr_val


# ─────────────────────────────────────────────────────────
# ORDER BLOCK DETECTION
# ─────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan OHLCV data and return a DataFrame of all detected Order Blocks.

    Columns:
      ob_type      : 'bullish' or 'bearish'
      ob_high      : top of the OB zone
      ob_low       : bottom of the OB zone
      ob_mid       : 50% level of the OB zone
      ob_size      : zone height
      ob_size_pct  : zone height as % of close price
      formed_at    : timestamp of the OB candle
      candle_idx   : integer position in df
      body_size    : absolute body size of the OB candle
      status       : 'fresh' | 'tested' | 'mitigated'
      first_test   : timestamp when first tested (or NaT)
      mitigation   : timestamp when mitigated (or NaT)
    """
    atr     = compute_atr(df)
    records = []

    for i in range(1, len(df) - IMPULSE_LOOKFORWARD):
        row     = df.iloc[i]
        atr_val = float(atr.iloc[i])
        if atr_val == 0:
            continue

        # ── Bullish OB: bearish candle followed by bullish impulse
        if is_bearish(row):
            if detect_bullish_impulse(df, i, atr):
                ob_high = float(row["High"])
                ob_low  = float(row["Low"])
                ob_size = ob_high - ob_low
                if ob_size >= MIN_OB_SIZE_ATR * atr_val:
                    records.append({
                        "ob_type":     "bullish",
                        "ob_high":     ob_high,
                        "ob_low":      ob_low,
                        "ob_mid":      (ob_high + ob_low) / 2,
                        "ob_size":     ob_size,
                        "ob_size_pct": ob_size / float(row["Close"]) * 100,
                        "formed_at":   df.index[i],
                        "candle_idx":  i,
                        "body_size":   candle_body_size(row),
                        "status":      "fresh",
                        "first_test":  pd.NaT,
                        "mitigation":  pd.NaT,
                    })

        # ── Bearish OB: bullish candle followed by bearish impulse
        elif is_bullish(row):
            if detect_bearish_impulse(df, i, atr):
                ob_high = float(row["High"])
                ob_low  = float(row["Low"])
                ob_size = ob_high - ob_low
                if ob_size >= MIN_OB_SIZE_ATR * atr_val:
                    records.append({
                        "ob_type":     "bearish",
                        "ob_high":     ob_high,
                        "ob_low":      ob_low,
                        "ob_mid":      (ob_high + ob_low) / 2,
                        "ob_size":     ob_size,
                        "ob_size_pct": ob_size / float(row["Close"]) * 100,
                        "formed_at":   df.index[i],
                        "candle_idx":  i,
                        "body_size":   candle_body_size(row),
                        "status":      "fresh",
                        "first_test":  pd.NaT,
                        "mitigation":  pd.NaT,
                    })

    obs = pd.DataFrame(records)
    if obs.empty:
        return obs

    # Cap to most recent OBs per side
    bull_obs = obs[obs["ob_type"] == "bullish"].tail(MAX_OBS_PER_SIDE)
    bear_obs = obs[obs["ob_type"] == "bearish"].tail(MAX_OBS_PER_SIDE)
    return pd.concat([bull_obs, bear_obs]).sort_values("candle_idx").reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# STATUS TRACKING
# ─────────────────────────────────────────────────────────

def update_ob_status(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """
    For each OB, scan all candles AFTER formation and update:
      - status: fresh → tested → mitigated
      - first_test: when price first entered the zone
      - mitigation: when price closed through the zone

    Bullish OB mitigation : close < ob_low
    Bearish OB mitigation : close > ob_high
    Bullish OB test       : low <= ob_high (price touched the zone)
    Bearish OB test       : high >= ob_low (price touched the zone)
    """
    obs = obs.copy()

    for idx, ob in obs.iterrows():
        ci        = int(ob["candle_idx"])
        future_df = df.iloc[ci + 1 :]
        status    = "fresh"
        first_test = pd.NaT
        mitigation = pd.NaT

        for ts, row in future_df.iterrows():
            lo = float(row["Low"])
            hi = float(row["High"])
            cl = float(row["Close"])

            if ob["ob_type"] == "bullish":
                touched = lo <= float(ob["ob_high"])
                mitig   = cl < float(ob["ob_low"])
            else:
                touched = hi >= float(ob["ob_low"])
                mitig   = cl > float(ob["ob_high"])

            if touched and status == "fresh":
                status     = "tested"
                first_test = ts

            if mitig:
                status     = "mitigated"
                mitigation = ts
                break

        obs.at[idx, "status"]     = status
        obs.at[idx, "first_test"] = first_test
        obs.at[idx, "mitigation"] = mitigation

    return obs


# ─────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────

def compute_stats(obs: pd.DataFrame) -> dict:
    stats = {}
    total = len(obs)
    if total == 0:
        return stats

    for ob_type in ["bullish", "bearish", "all"]:
        sub = obs if ob_type == "all" else obs[obs["ob_type"] == ob_type]
        if sub.empty:
            continue

        n          = len(sub)
        n_fresh    = (sub["status"] == "fresh").sum()
        n_tested   = (sub["status"] == "tested").sum()
        n_mitig    = (sub["status"] == "mitigated").sum()

        # Candles between formation and first test
        def candles_to_event(ob_row, event_ts):
            if pd.isna(event_ts):
                return np.nan
            ci    = int(ob_row["candle_idx"])
            try:
                ei = df_ref.index.get_loc(event_ts)
            except Exception:
                return np.nan
            return max(0, ei - ci)

        stats[ob_type] = {
            "count":         n,
            "fresh_pct":     n_fresh  / n * 100,
            "tested_pct":    n_tested / n * 100,
            "mitig_pct":     n_mitig  / n * 100,
            "avg_size_pct":  sub["ob_size_pct"].mean(),
            "median_size_pct": sub["ob_size_pct"].median(),
        }

    return stats


# ─────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────

def print_report(ticker: str, tf_label: str, obs: pd.DataFrame):
    W = 66
    print()
    print("═" * W)
    print(f"  {ticker}  |  {tf_label}  |  {len(obs)} Order Blocks detected")
    print("═" * W)

    for ob_type in ["bullish", "bearish"]:
        sub = obs[obs["ob_type"] == ob_type]
        if sub.empty:
            continue
        n       = len(sub)
        fresh   = (sub["status"] == "fresh").sum()
        tested  = (sub["status"] == "tested").sum()
        mitig   = (sub["status"] == "mitigated").sum()
        sym     = "↑" if ob_type == "bullish" else "↓"

        print(f"\n  {sym} {ob_type.upper()} OBs  (n={n})")
        print(f"  {'Status':<14} {'Count':>6}  {'Pct':>7}  Bar")
        print("  " + "─" * 46)
        for label, count in [("Fresh", fresh), ("Tested", tested), ("Mitigated", mitig)]:
            pct = count / n * 100 if n else 0
            bar = "█" * int(pct / 5)
            print(f"  {label:<14} {count:>6}  {pct:>6.1f}%  {bar}")

        print(f"\n  Avg OB size : {sub['ob_size_pct'].mean():.3f}% of price")

        # Show 5 most recent OBs
        recent = sub.tail(5).iloc[::-1]
        print(f"\n  Recent OBs (newest first):")
        print(f"  {'Formed':<22} {'High':>10} {'Low':>10} {'Status':<12}")
        print("  " + "─" * 56)
        for _, row in recent.iterrows():
            ts = str(row["formed_at"])[:16]
            print(f"  {ts:<22} {row['ob_high']:>10.4f} {row['ob_low']:>10.4f} {row['status']:<12}")

    print("\n" + "═" * W)


# ─────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────

def plot_order_blocks(df: pd.DataFrame, obs: pd.DataFrame,
                      ticker: str, tf_label: str,
                      output_path: str = "order_blocks.png",
                      lookback: int = 120):
    """
    Candlestick chart with OB zones overlaid as shaded rectangles.
    Only shows the most recent `lookback` candles for clarity.
    OB zones extend forward from formation to the right edge of the chart.
    """
    plot_df = df.tail(lookback).copy()
    if plot_df.empty:
        return

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[3, 1],
                            hspace=0.08, wspace=0.28)

    # ── Main candlestick panel (left, spans both rows) ──
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_facecolor(PANEL)
    for spine in ax_main.spines.values():
        spine.set_edgecolor(GRID_C)
    ax_main.grid(color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.6)
    ax_main.tick_params(colors=MUTED_C, labelsize=7.5)

    # Draw candles manually
    opens  = plot_df["Open"].values
    highs  = plot_df["High"].values
    lows   = plot_df["Low"].values
    closes = plot_df["Close"].values
    xs     = np.arange(len(plot_df))

    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = UP_CANDLE if c >= o else DOWN_CANDLE
        # Wick
        ax_main.plot([i, i], [l, h], color=color, linewidth=0.7, alpha=0.8)
        # Body
        body_lo = min(o, c)
        body_hi = max(o, c)
        ax_main.add_patch(mpatches.FancyBboxPatch(
            (i - 0.3, body_lo), 0.6, max(body_hi - body_lo, 0.0001),
            boxstyle="square,pad=0",
            facecolor=color, edgecolor=color, linewidth=0, alpha=0.9
        ))

    # Draw OB zones
    first_ts = plot_df.index[0]
    last_x   = len(plot_df) - 1

    for _, ob in obs.iterrows():
        if ob["formed_at"] < first_ts:
            # Zone started before plot window — extend from x=0
            x_start = 0
        else:
            try:
                x_start = plot_df.index.get_loc(ob["formed_at"])
            except KeyError:
                # find nearest
                pos = plot_df.index.searchsorted(ob["formed_at"])
                x_start = min(pos, last_x)

        if x_start > last_x:
            continue

        is_bull = ob["ob_type"] == "bullish"
        fill    = BULL_FILL if is_bull else BEAR_FILL
        edge    = BULL_EDGE if is_bull else BEAR_EDGE

        status  = ob["status"]
        alpha   = (FRESH_ALPHA     if status == "fresh"
                   else TESTED_ALPHA  if status == "tested"
                   else MITIGATED_ALPHA)

        width   = last_x - x_start + 1
        height  = float(ob["ob_high"]) - float(ob["ob_low"])

        ax_main.add_patch(mpatches.FancyBboxPatch(
            (x_start, float(ob["ob_low"])), width, height,
            boxstyle="square,pad=0",
            facecolor=fill, edgecolor=edge,
            linewidth=0.6, alpha=alpha
        ))

        # Mid-line
        mid = float(ob["ob_mid"])
        ax_main.hlines(mid, x_start, last_x,
                       colors=edge, linewidths=0.5,
                       linestyles="--", alpha=0.5)

        # Label (fresh OBs only, avoid clutter)
        if status == "fresh":
            label = "Bull OB" if is_bull else "Bear OB"
            ax_main.text(x_start + 0.5, float(ob["ob_high"]) + height * 0.08,
                         label, fontsize=6.5, color=edge, alpha=0.9)

    ax_main.set_xlim(-1, last_x + 2)
    price_range = plot_df["High"].max() - plot_df["Low"].min()
    ax_main.set_ylim(plot_df["Low"].min()  - price_range * 0.03,
                     plot_df["High"].max() + price_range * 0.05)

    # x-axis: show dates every ~20 candles
    step    = max(1, len(plot_df) // 6)
    x_ticks = xs[::step]
    x_labels = [str(plot_df.index[i])[:10] for i in x_ticks]
    ax_main.set_xticks(x_ticks)
    ax_main.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
    ax_main.set_title(f"{ticker}  ·  {tf_label}  ·  Order Blocks",
                      color=TEXT_C, fontsize=11, pad=8)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=BULL_FILL, edgecolor=BULL_EDGE,
                       alpha=0.6, label="Bullish OB"),
        mpatches.Patch(facecolor=BEAR_FILL, edgecolor=BEAR_EDGE,
                       alpha=0.6, label="Bearish OB"),
        Line2D([0], [0], color=BULL_EDGE, lw=0.8, ls="--", label="OB Mid (bull)"),
        Line2D([0], [0], color=BEAR_EDGE, lw=0.8, ls="--", label="OB Mid (bear)"),
    ]
    ax_main.legend(handles=legend_elements, fontsize=7.5,
                   facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C,
                   loc="upper left")

    # ── Status pie chart (top-right) ──
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_pie.set_facecolor(PANEL)
    for spine in ax_pie.spines.values():
        spine.set_edgecolor(GRID_C)

    n_fresh  = (obs["status"] == "fresh").sum()
    n_tested = (obs["status"] == "tested").sum()
    n_mitig  = (obs["status"] == "mitigated").sum()
    sizes    = [n_fresh, n_tested, n_mitig]
    labels   = [f"Fresh\n{n_fresh}", f"Tested\n{n_tested}", f"Mitigated\n{n_mitig}"]
    colors   = ["#a78bfa", "#38bdf8", MUTED_C]
    explode  = [0.05, 0.05, 0.02]

    wedges, texts = ax_pie.pie(
        [max(s, 0.001) for s in sizes],
        labels=labels, colors=colors, explode=explode,
        startangle=90, textprops={"fontsize": 8, "color": TEXT_C},
        wedgeprops={"edgecolor": BG, "linewidth": 1.5}
    )
    ax_pie.set_title("OB Status Breakdown", color=TEXT_C, fontsize=9, pad=6)

    # ── Size distribution bar chart (bottom-right) ──
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor(PANEL)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(GRID_C)
    ax_bar.grid(color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.6)
    ax_bar.tick_params(colors=MUTED_C, labelsize=7.5)

    bull_sizes = obs[obs["ob_type"] == "bullish"]["ob_size_pct"].values
    bear_sizes = obs[obs["ob_type"] == "bearish"]["ob_size_pct"].values

    all_sizes = np.concatenate([bull_sizes, bear_sizes])
    if len(all_sizes) > 0:
        bins = np.linspace(0, np.percentile(all_sizes, 95), 20)
        ax_bar.hist(bull_sizes, bins=bins, color=BULL_FILL, alpha=0.7,
                    label="Bullish", edgecolor=BG)
        ax_bar.hist(bear_sizes, bins=bins, color=BEAR_FILL, alpha=0.7,
                    label="Bearish", edgecolor=BG)
        ax_bar.set_xlabel("OB Size (% of price)", fontsize=8, color=MUTED_C)
        ax_bar.set_ylabel("Count", fontsize=8, color=MUTED_C)
        ax_bar.set_title("OB Size Distribution", color=TEXT_C, fontsize=9, pad=6)
        ax_bar.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C)
        ax_bar.xaxis.label.set_color(MUTED_C)
        ax_bar.yaxis.label.set_color(MUTED_C)

    fig.suptitle(
        f"ICT Order Block Analysis  ·  {ticker}  ·  {tf_label}  "
        f"·  {len(obs)} OBs detected",
        color=TEXT_C, fontsize=12, y=1.01
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  Chart saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────

def export_csv(obs: pd.DataFrame, path: str = "order_blocks.csv"):
    obs.to_csv(path, index=False)
    print(f"  CSV saved   → {path}  ({len(obs)} order blocks)")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

# Global reference for status computation (yfinance uses global df)
df_ref = None


def run_single(ticker: str, interval: str, tf_label: str,
               no_chart: bool, no_csv: bool):
    global df_ref

    print(f"\n  Fetching {ticker} [{tf_label}] ...", end=" ", flush=True)
    df = fetch_ohlcv(ticker, interval,
                     TIMEFRAME_MAP.get(interval, {}).get("period", "180d"))

    if interval == "4h":
        df = fetch_ohlcv(ticker, "1h",
                         TIMEFRAME_MAP["4h"]["period"])
        df = resample_4h(df)

    if df.empty or len(df) < 30:
        print("insufficient data, skipping.")
        return None, None

    print(f"{len(df):,} candles", end=" → ", flush=True)
    df_ref = df

    obs = detect_order_blocks(df)
    if obs.empty:
        print("no OBs detected.")
        return df, obs

    print(f"{len(obs)} OBs", end=" → ", flush=True)
    obs = update_ob_status(df, obs)
    print("status updated ✓")

    print_report(ticker, tf_label, obs)

    safe_ticker = ticker.replace("=", "").replace("/", "")
    if not no_chart:
        chart_path = f"order_blocks_{safe_ticker}_{tf_label}.png"
        plot_order_blocks(df, obs, ticker, tf_label, output_path=chart_path)

    if not no_csv:
        csv_path = f"order_blocks_{safe_ticker}_{tf_label}.csv"
        export_csv(obs, csv_path)

    return df, obs


def main():
    parser = argparse.ArgumentParser(
        description="ICT Order Block Detector & Status Tracker"
    )
    parser.add_argument("--ticker",    type=str, default=None,
                        help="Ticker symbol (e.g. AAPL, NQ=F, EURUSD=X)")
    parser.add_argument("--tf",        type=str, default="1h",
                        choices=["1h", "4h", "1d"],
                        help="Timeframe: 1h, 4h, 1d  (default: 1h)")
    parser.add_argument("--no-chart",  action="store_true",
                        help="Skip chart output")
    parser.add_argument("--no-csv",    action="store_true",
                        help="Skip CSV export")
    parser.add_argument("--lookback",  type=int, default=120,
                        help="Candles to show on chart (default: 120)")
    args = parser.parse_args()

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║      ICT Order Block Detector & Status Tracker        ║")
    print("╚════════════════════════════════════════════════════════╝")

    tf_label = TIMEFRAME_MAP.get(args.tf, {}).get("label", args.tf.upper())

    if args.ticker:
        run_single(args.ticker, args.tf, tf_label, args.no_chart, args.no_csv)
    else:
        print(f"  Running default tickers: {', '.join(DEFAULT_TICKERS)}")
        for ticker in DEFAULT_TICKERS:
            run_single(ticker, args.tf, tf_label, args.no_chart, args.no_csv)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()