# Workflow: US Equities Strategies with Backtrader + ML

Use this when designing, validating, and iterating ML-driven strategies for US stocks.

## Data + Labeling
- Prefer robust daily data (split/dividend adjusted). Default source in the template is Yahoo Finance via yfinance; swap in premium feeds when needed.
- Clean: forward-fill corporate action gaps only if aligned with adjusted data; drop obvious bad ticks; keep timezone consistent.
- Label: use next-period forward returns (e.g., 1d) to avoid leakage. For multi-horizon ideas (5d/10d), shift labels by that horizon and adjust signal alignment in Backtrader.
- Universe: start with 1-3 liquid tickers for speed, then expand. Keep delisted names out unless you have survivorship-bias-corrected data.

## Feature Engineering
- Keep early iterations simple and cheap: rolling returns, moving-average spreads, volatility, RSI, volume z-scores. Add market-regime signals (broad index trend) as needed.
- Normalize features (z-scores or scaling) for most linear models. Avoid lookahead by using only information available at time t.
- Drop NaNs after computing rolling windows to ensure clean alignment with labels.

## Modeling (Leakage-Safe)
- Fit models outside the Backtrader strategy; feed only predictions/signals into the engine.
- Use walk-forward training: expanding window with periodic refits (e.g., every 20 bars) to mimic live deployment.
- Start with simple, interpretable models (logistic regression) before tree ensembles. Guard against class imbalance (class weights or threshold tuning).
- Evaluate calibration (probabilities make thresholding easier) and sensitivity to retrain frequency.

## Backtesting in Backtrader
- Feed predictions as a custom `signal` column (see `SignalData` in the script). Shift signals by the label horizon so trades only use prior predictions.
- Add analyzers: `Returns`, `SharpeRatio`, `TradeAnalyzer`, `DrawDown`. Track turnover to estimate costs realistically.
- Position sizing: begin with 1-share stake or % of cash; add stop-loss/take-profit only after basic signal quality is validated.
- Costs/slippage: set realistic `commission` (bps) and optionally slippage models; re-run when adjusting thresholds.

## Evaluation Loop
1) Inspect raw signal hit-rate and confusion matrix on the walk-forward predictions before trading.
2) Backtest with conservative sizing and realistic costs.
3) Review drawdowns, exposure, turnover, and period-by-period returns (stability matters more than single-period gains).
4) Stress test: vary threshold, retrain frequency, and horizon; ensure results are not brittle.
5) Expand universe gradually; avoid mixing symbols in one cerebro run unless you explicitly handle multi-data strategies.

## Common Pitfalls to Avoid
- Lookahead: never fit on or use features that incorporate future bars; ensure labels are shifted forward.
- Data snooping: too many handcrafted features without hypothesis; prefer small, justified sets.
- Overfitting thresholds: choose thresholds using a validation slice, not the full backtest period.
- Ignoring trading frictions: include commissions/slippage early; US equities are sensitive when turnover is high.
