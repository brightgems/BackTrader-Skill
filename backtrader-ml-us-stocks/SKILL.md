---
name: backtrader-ml-us-stocks
description: Build and iterate US equities trading strategies using Backtrader plus machine learning. Use when asked to design, prototype, backtest, or tune algorithmic strategies for US stocks with Backtrader, walk-forward ML signals, feature engineering, and leakage-safe evaluation.
---

# Backtrader ML US Stocks

## Overview

End-to-end guide for designing ML-driven US equity strategies: fetch data, engineer features, train walk-forward models, pipe predictions into Backtrader, and evaluate with realistic costs and guardrails.

## Quick Start
- If you need a runnable scaffold, adapt `scripts/backtrader_ml_template.py` (`--data-source` supports yfinance or akshare; `--proxy` lets yfinance use an HTTP(S) proxy such as `http://localhost:7890`; defaults to auto-fallback). Features include MA spreads, vol, RSI, volume z-score, MACD hist, ATR-based risk, momentum z-score, and optional beta vs a benchmark (`--bench-ticker`, default SPY). Models: logistic (default), RandomForest (`--model random_forest` with `--n-estimators/--max-depth`), or hist GBDT (`--model hist_gbdt` with `--gbdt-estimators/--gbdt-depth/--gbdt-lr`).
- For workflow details and guardrails, load `references/workflow.md`.
- For additional feature/label ideas, see `references/feature-ideas.md`.

## Core Workflow
1) Scope the idea: target horizon (1d/5d), rough holding period, turnover tolerance, and cost assumptions (bps).
2) Data + labeling: pull adjusted OHLCV (yfinance in template); compute forward-return labels shifted by horizon; drop NaNs from rolling features.
3) Modeling: train outside Backtrader. Use walk-forward fitting (expanding window, periodic refits) to avoid leakage; start with logistic regression before heavier models.
4) Convert predictions to signals: store predicted probability of up-move; choose threshold (e.g., 0.55) based on validation, not on the full backtest.
5) Backtest in Backtrader: feed a `signal` column (see `SignalData`); set cash, commission/slippage; add analyzers (`Returns`, `SharpeRatio`, `DrawDown`).
6) Evaluate + iterate: inspect precision/recall of signals, drawdowns, turnover, cost sensitivity, and robustness to threshold/retrain-frequency changes.

## Resources (optional)

- `scripts/backtrader_ml_template.py`: end-to-end sample (yfinance fetch with akshare fallback via `--data-source`, feature creation, walk-forward logistic model, Backtrader backtest). Adjust features/label horizon/thresholds to match your idea.
- `references/workflow.md`: detailed, leakage-safe workflow and evaluation checklist for Backtrader + ML on US equities.
- `references/feature-ideas.md`: small menu of features/labels/validation variations to try when iterating.
