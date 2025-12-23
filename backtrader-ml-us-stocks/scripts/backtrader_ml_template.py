"""
Lightweight Backtrader + ML template for US equities.

Dependencies (install as needed): backtrader, pandas, numpy, yfinance, scikit-learn.
This script is meant as a starting point—adapt features, labeling, and execution rules
to your specific strategy idea.
"""

import argparse
import datetime as dt
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import backtrader as bt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import yfinance as yf
import os


def fetch_ohlcv(
    ticker: str, start: str, end: str, source: str = "auto", proxy: str | None = None
) -> pd.DataFrame:
    """
    Download US equity OHLCV data. Default tries yfinance, then akshare as fallback.
    """
    last_err = None
    if source in ("auto", "yfinance"):
        try:
            if proxy:
                os.environ["HTTP_PROXY"] = proxy
                os.environ["HTTPS_PROXY"] = proxy
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    col_levels = list(data.columns.names)
                    if "Ticker" in col_levels:
                        data = data.xs(ticker, axis=1, level="Ticker")
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                    elif ticker in data.columns.get_level_values(0):
                        data = data.xs(ticker, axis=1, level=0)
                    else:
                        data = data.droplevel(0, axis=1)
                data.index = pd.to_datetime(data.index)
                return data
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
    if source in ("auto", "akshare"):
        try:
            import akshare as ak

            start_int = pd.to_datetime(start).strftime("%Y%m%d")
            end_int = pd.to_datetime(end).strftime("%Y%m%d")
            df = ak.stock_us_daily(symbol=ticker.upper(), adjust="")
            if df.empty:
                raise ValueError("akshare returned no data")
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_int) & (df["date"] <= end_int)]
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            ).set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
            # akshare lacks adjusted close; use close as proxy
            df["Adj Close"] = df["Close"]
            return df
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
    raise ValueError(f"No data returned for {ticker}. Last error: {last_err}")


def macd_hist(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal


def rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = asset_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    return cov / var


def average_true_range(data: pd.DataFrame, price_col: str, period: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    prev_close = data[price_col].shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple, fast-to-compute features from OHLCV data."""
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[price_col]
    returns = prices.pct_change()

    feats = pd.DataFrame(index=data.index)
    feats["ret_1"] = returns
    feats["ret_5"] = prices.pct_change(5)
    feats["sma_10"] = prices.rolling(10).mean() / prices - 1
    feats["sma_20"] = prices.rolling(20).mean() / prices - 1
    feats["vol_10"] = returns.rolling(10).std()
    feats["vol_20"] = returns.rolling(20).std()
    feats["rsi_14"] = compute_rsi(prices, window=14)
    feats["volume_z"] = (data["Volume"] - data["Volume"].rolling(20).mean()) / data[
        "Volume"
    ].rolling(20).std()
    feats["macd_hist"] = macd_hist(prices)
    feats["atr_14"] = average_true_range(data, price_col, period=14) / prices
    mom_20 = prices.pct_change(20)
    feats["mom_z_20"] = (mom_20 - mom_20.rolling(120).mean()) / mom_20.rolling(120).std()
    if "bench_ret" in data.columns:
        feats["beta_60"] = rolling_beta(returns, data["bench_ret"], window=60)

    return feats.dropna()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def label_forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Label next-period direction to avoid lookahead.
    Returns 1 if forward return > 0, else 0.
    """
    fwd_return = prices.shift(-horizon) / prices - 1
    return (fwd_return > 0).astype(int)


def walk_forward_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    model_factory: Callable[[], Pipeline],
    warmup: int = 150,
    retrain_every: int = 20,
) -> pd.Series:
    """
    Rolling walk-forward training to reduce leakage.
    Fits on data up to t-1, predicts at t.
    """
    if len(features) <= warmup:
        raise ValueError("Not enough data for warmup window.")

    preds = []
    index = features.index
    for i in range(warmup, len(features)):
        if (i - warmup) % retrain_every == 0:
            model = model_factory()
            model.fit(features.iloc[:i], labels.iloc[:i].values.ravel())
        prob_up = model.predict_proba(features.iloc[[i]])[0, 1]
        preds.append(prob_up)
    pred_index = index[warmup:]
    return pd.Series(preds, index=pred_index, name="signal_prob")


class SignalData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (("signal", -1),)


class MLSignalStrategy(bt.Strategy):
    params = dict(threshold=0.55, stake=1.0)

    def __init__(self):
        self.signal = self.datas[0].signal

    def next(self):
        sig = self.signal[0]
        if not self.position and sig > self.p.threshold:
            self.buy(size=self.p.stake)
        elif self.position and sig <= 0.5:
            self.close()


def build_signal_frame(
    data: pd.DataFrame, signal: pd.Series, horizon: int
) -> pd.DataFrame:
    """
    Align predictions with OHLCV; shift by horizon to ensure trade uses prior prediction.
    """
    df = data.copy()
    df["signal"] = signal.reindex(df.index).shift(horizon)
    return df.dropna()


def run_backtest(
    signal_df: pd.DataFrame, cash: float, commission: float, threshold: float
) -> Tuple[bt.Cerebro, bt.Strategy]:
    cerebro = bt.Cerebro()
    data = SignalData(dataname=signal_df)
    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addstrategy(MLSignalStrategy, threshold=threshold, stake=1.0)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    return cerebro, cerebro.run()


def summarize_results(cerebro: bt.Cerebro, strat):
    ending_value = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
    returns = strat.analyzers.returns.get_analysis()
    print(f"Ending value: {ending_value:,.2f}")
    if sharpe is not None:
        print(f"Daily Sharpe: {sharpe:.2f}")
    print(f"Total return: {returns.get('rtot', 0):.2%}")
    print(f"Annual return: {returns.get('rnorm', 0):.2%}")


def default_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", C=1.0, max_iter=200, solver="lbfgs", n_jobs=1
                ),
            ),
        ]
    )


def rf_model(n_estimators: int = 200, max_depth: int | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def hgb_model(max_depth: int | None = None, learning_rate: float = 0.05, max_iter: int = 300) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backtrader walk-forward ML demo for US equities."
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=dt.date.today().isoformat())
    parser.add_argument("--cash", type=float, default=100000)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--horizon", type=int, default=1, help="Forward return horizon for labeling.")
    parser.add_argument("--warmup", type=int, default=250, help="Samples before first prediction.")
    parser.add_argument("--retrain", type=int, default=20, help="Retrain frequency for walk-forward.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Buy threshold on predicted up probability.")
    parser.add_argument("--data-source", choices=["auto", "yfinance", "akshare"], default="auto", help="Data source preference.")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP(S) proxy URL for yfinance (e.g., http://localhost:7890).")
    parser.add_argument("--bench-ticker", type=str, default="SPY", help="Benchmark ticker for beta feature; blank to disable.")
    parser.add_argument("--model", choices=["logistic", "random_forest", "hist_gbdt"], default="logistic", help="Model type for signal generation.")
    parser.add_argument("--n-estimators", type=int, default=200, help="RandomForest trees when --model=random_forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="RandomForest max depth when --model=random_forest.")
    parser.add_argument("--gbdt-depth", type=int, default=None, help="Max depth for hist GBDT.")
    parser.add_argument("--gbdt-lr", type=float, default=0.05, help="Learning rate for hist GBDT.")
    parser.add_argument("--gbdt-estimators", type=int, default=300, help="Iterations for hist GBDT.")
    args = parser.parse_args()

    raw = fetch_ohlcv(args.ticker, args.start, args.end, source=args.data_source, proxy=args.proxy)
    if args.bench_ticker:
        bench_df = fetch_ohlcv(args.bench_ticker, args.start, args.end, source=args.data_source, proxy=args.proxy)
        bench_price_col = "Adj Close" if "Adj Close" in bench_df.columns else "Close"
        raw["bench_ret"] = bench_df[bench_price_col].pct_change().reindex(raw.index)
    feats = engineer_features(raw)
    labels = label_forward_returns(raw["Adj Close"], horizon=args.horizon).reindex(feats.index)
    labels = labels.dropna()
    feats = feats.loc[labels.index]

    if args.model == "random_forest":
        model_factory = lambda: rf_model(args.n_estimators, args.max_depth)
    elif args.model == "hist_gbdt":
        model_factory = lambda: hgb_model(args.gbdt_depth, args.gbdt_lr, args.gbdt_estimators)
    else:
        model_factory = default_model

    preds = walk_forward_predictions(
        feats, labels, model_factory=model_factory, warmup=args.warmup, retrain_every=args.retrain
    )
    signal_frame = build_signal_frame(raw, preds, horizon=args.horizon)

    cerebro, strategies = run_backtest(
        signal_frame, cash=args.cash, commission=args.commission, threshold=args.threshold
    )
    strat = strategies[0]
    summarize_results(cerebro, strat)
    # Uncomment to visualize:
    # cerebro.plot()


if __name__ == "__main__":
    main()
