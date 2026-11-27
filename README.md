# AGENT 1 — Market-Structure Scanner & Closed-Loop MLOps Desktop System

*Private client build — source code is kept in a private repository to protect client IP. This repository documents the system architecture, ML evaluation methodology, production debugging lessons, and a live walkthrough.*

---

## Live Demo

<!-- Drag and drop your 60–90s MP4 recording directly into the GitHub web editor here so it renders natively inline -->
> **[Demo video placeholder — replace this line by dragging your `.mp4` screen recording into GitHub's editor]**
>
> *Recommended 60–90s walkthrough: (1) Launch `AGENT 1` from Desktop and watch auto-armed live TradingView charts load across 7 markets, (2) Switch timeframes (`1D → 4H → 1H → 15m → 5m`) and scroll left to show on-demand historical bar loading, (3) Open `Settings → 2-Year Backtest` to show the 10,253-setup & 300-trade self-training summary, (4) Trigger `Test Sound` to demonstrate the 5-field alert notification + 2-minute alert ringtone.*

---

## Why This Was Built

Most retail trading tools break down in one of two ways: either they stay trapped inside a Jupyter notebook that never survives live market data, or they ship as noisy indicator scripts that flood the screen with dozens of overlapping zones and false alerts.

**AGENT 1** was built for a discretionary trader who trades multi-timeframe Smart Money Concepts (SMC) across seven instruments (`XAU/USD`, `XAG/USD`, `NAS100`, `UK100`, `EUR/USD`, `GBP/JPY`, `GBP/USD`). The requirements were strict and practical:

- **Zero setup on the client's machine:** A standalone Windows desktop installer (`.exe`) — no Python, Node.js, Docker, or API keys required.
- **Real market data only:** Live TradingView WebSocket feeds with deep scrollable history (`1,000+` bars per timeframe plus on-demand left-scroll backfill) and zero synthetic fallback candles.
- **Simultaneous Intraday + Scalping scanning:** Top-down structure analysis running continuously across all seven markets without forcing manual mode switches.
- **Asymmetric risk-to-reward gating (`1:3` to `1:15` R:R):** Targets projected to actual opposing higher-timeframe liquidity levels rather than fixed static multiples.
- **Low-noise delivery:** Clean, decluttered chart markups around current price, and a minimalist 5-field alert card (`Asset`, `BUY/SELL`, `Entry`, red `Stop Loss`, green `Take Profit`) backed by a 2-minute continuous alert ringtone.

---

## System Architecture

```text
Offline Training & Self-Correction Pipeline
───────────────────────────────────────────
  49 Mentor Strategy Videos (transcript extraction -> structured rule priors)
  + 2-Year Historical OHLCV (Yahoo Finance 1H 2yr + 15m 60d across 7 assets)
        │
        ▼
  Deterministic Setup Generator (10,253 setups, 1:3–1:15 R:R, strict SL-first labeling)
        │
        ▼
  22 Leakage-Free Features -> 5-Fold Chronological TimeSeriesSplit LightGBM
        │
        ▼
  3x300 Walk-Forward Position-Block Simulator (900 simulated R:R blocks)
  -> Error diagnosis by regime/style/R:R -> Hard-negative & timeout-denoised retraining
        │
        ▼
  4 Shadow Validation Gates (0 NaNs | min rows | CV AUC floor | ONNX parity < 1e-5)
        │
        ▼
  Exported ONNX Artifact (63.6 KB, opset 14, < 0.3 ms CPU inference)

Online Production Desktop Runtime (Windows NSIS Installer — 200.8 MB)
─────────────────────────────────────────────────────────────────────
  TradingView WSS (single multiplexed socket, 7 assets x 7 timeframes)
        │
        ▼
  In-Memory Rolling Buffer (2,500 bars/TF + lazy /more left-scroll backfill)
        │
        ▼
  Unified Cascade Engine (Intraday 1D->4H->1H->30m->15m->5m/1m + Scalping 1H->15m->5m->1m)
        │
        ▼
  Dynamic R:R Gatekeeper (1:3–1:15) -> ONNX Scorer (A >= 0.50, A+ >= 0.60)
        │                                     │
        ▼                                     ▼
  Electron + React + Lightweight Charts   Outcome Tracker + PSI Drift Monitor
  (5-field alert card + 2m ringtone)      -> Isolated --train-worker subprocess
                                          -> Atomic hot-swap & 1-click rollback
```

---

## Engineering Problems Encountered & How They Were Solved

The interesting MLOps work was not training a classifier — it was debugging everything that breaks when a model leaves a notebook and runs on live feeds inside a packaged desktop app.

### 1. TradingView Rate Limiting (`HTTP 429`) and Empty Left-Scroll History
Opening one WebSocket per instrument immediately triggered `429` throttling from TradingView, and loading only 200 bars meant scrolling left hit a blank wall.

- **Fix:** Rewrote ingestion around a single persistent multiplexed WebSocket that pre-seeds all 7 assets across 7 timeframes sequentially using `create_series` / `modify_series` (`1,000+` bars per timeframe in ~6 seconds), and wired TradingView's `request_more_data` behind `GET /api/candles/{asset}/more`. When the chart's visible logical range approaches the left edge, older bars are prepended while shifting the logical range by the exact count of prepended bars so the viewport stays anchored.
- **No silent fallbacks:** If the socket drops, the UI shows a calm `Reconnecting` strip and backs off exponentially (`2s -> 60s`). It never generates synthetic candles to hide an outage.

### 2. Why ROC-AUC Alone Was Lying on Asymmetric `1:3`–`1:15` Payoffs
The first 2-year model achieved `0.6239` chronological 5-fold CV ROC-AUC and lifted out-of-sample win rate from `14.12%` to `20.67%` on the top 300 setups — yet its net return was **negative (`-23.69R`, Profit Factor `0.900`)**.

Looking at the trade distribution explained why immediately: `class_weight="balanced"` rewarded easy, low-variance `3.0–3.4R` targets that barely missed breakeven once spread and timeouts were accounted for, while penalizing wider `5R–12R` liquidity runs that actually drove expectancy.

- **Fix:** Built a chronological walk-forward simulator (`position_simulator.py`) that places 300 TradingView-style Long/Short R:R position blocks per round across unseen market windows (`900` total blocks across 3 rounds), breaks down performance by style, session, volatility regime, and R:R bucket, and retrains using shallower regularized trees (`max_depth=3`, higher L1/L2), timeout down-weighting (`0.7x`), and hard-negative up-weighting (`1.2x`), ranking candidates by expected return (`p * RR - (1 - p)`) rather than raw probability alone.
- **Result on 2-year strictly out-of-sample top-300 setups:**

| Model Version | OOF Top-300 Trades | Win Rate | Avg R:R | Net Return | Expectancy / Trade | Profit Factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline LightGBM (prob-only) | 300 | 20.67% | 1:3.40 | `-23.69R` | `-0.079R` | `0.900` |
| **AGENT 1 v3.0 (self-trained)** | **300** | **15.67%** | **1:7.68** | **`+90.84R`** | **`+0.303R`** | **`1.365`** |

- **Honest caveat on regime sensitivity:** Performance is not uniform across every window. In the most choppy sub-window (Round 3: May–Nov 2026), both models were negative — though v3.0 still cut drawdown relative to baseline (`-50.28R`, PF `0.805` vs. baseline `-55.82R`, PF `0.770`). On a separate 7-day replay of real TradingView candles across all 7 assets (`71` triggered alerts, `59` resolved), the production model produced `17` wins / `42` losses (`28.8%` resolved win rate, `+24.83R` net).

### 3. Why the First Live Version Stayed Silent During Big Moves
Early client testing surfaced a critical bug: the scanner showed active, major market moves happened, and zero alerts fired. Replaying live candles through the engine isolated four compounding bottlenecks:

1. **Startup state:** The backend booted in `IDLE` waiting for a manual button click instead of arming automatically.
2. **POI-tap vs. CHoCH timing paradox:** The cascade checked whether the *latest* candle was still inside the 15m/5m zone *after* a lower-timeframe displacement had already broken structure away from the zone — almost impossible simultaneously. Fixed by tracking recent zone taps over a bounded lookback window before CHoCH confirmation.
3. **Rigid spread and target caps:** A flat `3.0` pip spread ceiling was silently rejecting normal quotes on Gold, Silver, NAS100, and GBP/JPY, and target selection was collapsing toward `1:3.0` instead of scanning opposing HTF liquidity pools up to `1:15.0`.
4. **Stale model persistence across upgrades:** On Windows, runtime state lives in `%USERPROFILE%\.brad_trades_scanner\` so `Program Files` stays read-only. Re-installing a newer `.exe` kept loading the old `.onnx` copy cached in `AppData` until we added an `mtime` auto-upgrade check at startup.

### 4. Native Windows OpenMP Crashes During In-App Retraining
Running `onnxruntime` inference and `lightgbm` training inside the same long-running FastAPI process on Windows triggered native MSVC/OpenMP access violations (`0x0000000000000000`).

- **Fix:** Isolated retraining into a dedicated child process invoked via a `--train-worker` entrypoint inside the compiled PyInstaller executable. The worker pulls fresh market data, builds features, runs 5-fold chronological CV, exports ONNX, and must clear **4 shadow validation gates** (`0` NaNs across all 22 features, `>= 300` valid setups, CV ROC-AUC `>= 0.60`, and `max |LGBM - ONNX| < 1e-5`) before backing up the current model, atomically swapping the artifact, and calling `reload_model()` in-memory.

### 5. Clean-Machine Packaging Without Python or Node
Shipping a Python + ML + React stack to a non-technical Windows laptop exposed plenty of packaging footguns (`torch` and `matplotlib` pulled in as unused transitive dependencies, hardcoded relative paths breaking under `C:\Program Files`, Electron throttling background timers and muting audio when minimized).

- **Fixes:**
  - Excluded unused heavy modules in PyInstaller (`torch`, `torchvision`, `matplotlib`, `jupyter`, `duckdb`, `tkinter`), bringing the full NSIS installer (`AGENT 1 Setup.exe`) down to `200.81 MB`.
  - Configured Electron with `autoplay-policy=no-user-gesture-required`, `disable-background-media-suspend`, and `backgroundThrottling: false` so the 2-minute alert ringtone and Windows toast fire even when minimized to the system tray.
  - Verified every release build by launching the packaged `backend_server.exe` under a **stripped Windows `PATH` (`C:\Windows\System32;C:\Windows`)** with a fresh empty `APP_DATA_DIR`, confirming zero host Python/Node dependencies and zero orphaned processes on exit.

---

## Verified Model & System Telemetry

| Metric / Artifact | Verified Value |
| --- | --- |
| **Historical Dataset** | `10,253` setups (`2024-12-03` to `2026-11-23`), 7 instruments, real OHLCV only |
| **Feature Vector** | `22` normalized market-mechanics features, `0` NaNs / `0` Infs |
| **Outcome Labeling** | Strict conservative SL-before-TP precedence on every forward candle |
| **5-Fold Chronological CV ROC-AUC** | `0.6214` (folds: `0.6549`, `0.6600`, `0.6110`, `0.5965`, `0.5844`; train-val gap `0.0701`) |
| **Production ONNX Artifact** | `63,580` bytes, opset `14`, max abs diff vs. LightGBM = `1.42e-07`, `< 0.3 ms` CPU latency |
| **Top Predictive Features (by Gain)** | `tp_distance_atr`, `risk_reward_ratio`, `spread_to_sl_ratio`, `volatility_regime_ratio`, `hour_of_day_utc` |
| **Automated Test Suite** | `26/26` pytest passing + `8/8` end-to-end live audit checks + `tsc && vite build` (`0` errors) |
| **Clean-Machine Boot Time** | `6s` to `LIVE_CONNECTED` across all 7 pairs under stripped `PATH` |

---

## Known Limitations & What I Would Improve Next

1. **Probability calibration:** The production thresholds (`A >= 0.50`, `A+ >= 0.60`) act as calibrated ranking tiers rather than literal win-probability estimates; adding out-of-fold isotonic calibration would make expected-value position sizing cleaner.
2. **Execution slippage modeling:** Backtests use conservative SL-before-TP bar ordering and live spread filters, but true fill slippage during high-impact news spikes (CPI / NFP) is not yet modeled.
3. **Broader regime monitoring:** PSI feature drift and rolling live outcome tracking are wired and active, but meaningful live retraining triggers require accumulating at least 15–30 resolved live alerts per instrument across multiple sessions.

---

## Tech Stack

- **Backend & Ingestion:** Python 3.12, FastAPI, Uvicorn, Async WebSockets, Pydantic v2, Redis (with thread-safe in-memory fallback)
- **ML & MLOps:** LightGBM, ONNX / `onnxmltools`, ONNX Runtime, Scikit-learn (`TimeSeriesSplit`), MLflow, Prometheus, PSI Drift Detection, Pytest, GitHub Actions
- **Desktop & Frontend:** Electron 28, React 18, TypeScript, Tailwind CSS, TradingView Lightweight Charts, HTML5 Canvas Overlays, PyInstaller + Electron Builder (NSIS)
