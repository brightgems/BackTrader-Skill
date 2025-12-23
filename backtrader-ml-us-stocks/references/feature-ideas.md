# Feature and Label Playbook (US Equities)

Use selectively; keep the initial set small and expand only with clear hypotheses.

## Features
- Trend/mean-reversion: rolling returns (1/5/10/20d), MA spreads (10-20, 20-50), price vs. VWAP (if intraday available).
- Volatility/risk: rolling stdev of returns, ATR, realized beta vs. SPY/QQQ (requires benchmark series).
- Volume/flow: volume z-score vs. 20d mean, price change divided by volume, up/down volume ratio.
- Momentum quality: RSI(14), Stochastics %K/%D, MACD histogram (use as raw input, not signals).
- Seasonality/time: day-of-week dummy, month dummy; only if justified and tested for stability.
- Cross-sectional (when multi-asset): rank features within universe, but avoid mixing universes without robust data.

## Labels
- Directional: forward return > 0 over 1/5/10 bars (shifted).
- Magnitude: sign of forward return and bucketed strength (e.g., <-1%, -1% to 1%, >1%).
- Regime: market regime classification (bull/bear/sideways) inferred from index trend; use as conditioning feature, not target.

## Validation Variations
- Walk-forward with expanding window; re-train every N bars and vary N.
- Purged/embargoed splits if you hold trades across multiple bars.
- Threshold sweeps: measure precision/recall and turnover for thresholds (0.5-0.7) before deciding.
- Cost sensitivity: rerun with 0-10 bps to see robustness; high turnover strategies should be penalized.
