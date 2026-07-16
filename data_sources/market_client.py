"""
market_client.py — live/recent market data + technical indicators, computed
deterministically in Python (never by the LLM).

Priority order:
1. Finnhub for the latest quote.
2. yfinance for historical OHLCV data, technical indicators, and current-price
   fallback.

Finnhub and yfinance may fail independently.

All non-finite values (NaN, +inf, -inf) are converted to unavailable values
rather than being propagated into technical indicators or sent to the LLM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .cache import (
    TTL_HISTORICAL_MARKET,
    cache_key,
    get_cached_data,
    set_cached_data,
)
from .finnhub_client import finnhub_client
from .models import MarketDataSnapshot


logger = logging.getLogger("finbot.market")


# ---------------------------------------------------------------------------
# Numeric cleaning helpers
# ---------------------------------------------------------------------------

def _clean_float(value) -> Optional[float]:
    """
    Convert a scalar numeric value to a finite Python float.

    Returns None for:
    - None
    - NaN
    - +inf
    - -inf
    - non-numeric values
    """
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _clean_numeric_array(values) -> np.ndarray:
    """
    Convert values to a one-dimensional float array and remove all non-finite
    observations.
    """
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)

    return array[np.isfinite(array)]


# ---------------------------------------------------------------------------
# Deterministic technical calculations
# ---------------------------------------------------------------------------

def _pct_return(
    series: np.ndarray,
    periods_back: int,
) -> Optional[float]:
    """
    Calculate percentage return over an exact number of trading observations.

    Returns None when insufficient history exists. It does not silently shorten
    the requested return horizon.
    """
    series = _clean_numeric_array(series)

    if periods_back <= 0:
        return None

    if len(series) <= periods_back:
        return None

    start = _clean_float(series[-periods_back - 1])
    end = _clean_float(series[-1])

    if start in (None, 0) or end is None:
        return None

    result = (end / start - 1) * 100

    return (
        round(result, 2)
        if np.isfinite(result)
        else None
    )


def _sma(
    series: np.ndarray,
    window: int,
) -> Optional[float]:
    series = _clean_numeric_array(series)

    if len(series) < window:
        return None

    value = np.mean(series[-window:])

    return (
        round(float(value), 2)
        if np.isfinite(value)
        else None
    )


def _rsi(
    closes: np.ndarray,
    period: int = 14,
) -> Optional[float]:
    """
    Calculate a simple 14-period RSI using average gains and losses over the
    latest lookback window.
    """
    closes = _clean_numeric_array(closes)

    if len(closes) < period + 1:
        return None

    deltas = np.diff(
        closes[-(period + 1):]
    )

    gains = np.clip(
        deltas,
        0,
        None,
    )

    losses = -np.clip(
        deltas,
        None,
        0,
    )

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
        return None

    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0

        return 100.0

    rs = avg_gain / avg_loss

    value = 100 - (
        100 / (1 + rs)
    )

    return (
        round(float(value), 2)
        if np.isfinite(value)
        else None
    )


def _ema(
    series: np.ndarray,
    span: int,
) -> np.ndarray:
    """
    Deterministic exponential moving average.

    Input must already contain finite observations.
    """
    series = _clean_numeric_array(series)

    if len(series) == 0:
        return np.array([], dtype=float)

    alpha = 2 / (span + 1)

    ema = np.empty(
        len(series),
        dtype=float,
    )

    ema[0] = series[0]

    for i in range(1, len(series)):
        ema[i] = (
            alpha * series[i]
            + (1 - alpha) * ema[i - 1]
        )

    return ema


def _macd(
    closes: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate MACD(12, 26) and 9-period signal line.

    At least 35 finite closing-price observations are required.
    """
    closes = _clean_numeric_array(closes)

    if len(closes) < 35:
        return None, None

    ema12 = _ema(
        closes,
        12,
    )

    ema26 = _ema(
        closes,
        26,
    )

    if len(ema12) == 0 or len(ema26) == 0:
        return None, None

    macd_line = ema12 - ema26

    signal_line = _ema(
        macd_line,
        9,
    )

    if len(signal_line) == 0:
        return None, None

    macd_value = _clean_float(
        macd_line[-1]
    )

    signal_value = _clean_float(
        signal_line[-1]
    )

    return (
        round(macd_value, 3)
        if macd_value is not None
        else None,
        round(signal_value, 3)
        if signal_value is not None
        else None,
    )


def _historical_volatility(
    closes: np.ndarray,
) -> Optional[float]:
    """
    Calculate annualized historical volatility from daily log returns.
    """
    closes = _clean_numeric_array(closes)

    if len(closes) < 20:
        return None

    # Log returns require strictly positive prices.
    if np.any(closes <= 0):
        return None

    log_returns = np.diff(
        np.log(closes)
    )

    log_returns = _clean_numeric_array(
        log_returns
    )

    if len(log_returns) < 2:
        return None

    volatility = (
        np.std(log_returns)
        * np.sqrt(252)
        * 100
    )

    return (
        round(float(volatility), 2)
        if np.isfinite(volatility)
        else None
    )


def _max_drawdown(
    closes: np.ndarray,
) -> Optional[float]:
    """
    Calculate maximum peak-to-trough drawdown over the supplied price series.
    """
    closes = _clean_numeric_array(closes)

    if len(closes) < 2:
        return None

    if np.any(closes <= 0):
        return None

    running_max = np.maximum.accumulate(
        closes
    )

    drawdowns = (
        closes - running_max
    ) / running_max

    drawdowns = _clean_numeric_array(
        drawdowns
    )

    if len(drawdowns) == 0:
        return None

    value = np.min(
        drawdowns
    ) * 100

    return (
        round(float(value), 2)
        if np.isfinite(value)
        else None
    )


# ---------------------------------------------------------------------------
# Market data client
# ---------------------------------------------------------------------------

class MarketDataClient:

    def get_history(
        self,
        ticker: str,
        period: str = "2y",
    ):
        """
        Return a cleaned yfinance-style historical DataFrame.

        Two years are requested by default so the system has enough history for:
        - SMA200
        - approximately one-year returns
        - 52-week range
        - technical indicators

        Returns None on failure.
        """
        ticker = ticker.upper().strip()

        key = cache_key(
            "yfinance",
            "history",
            ticker,
            period,
        )

        cached = get_cached_data(
            key
        )

        if cached is not None:
            try:
                import pandas as pd

                df = pd.DataFrame(
                    cached["data"]
                )

                df.index = pd.to_datetime(
                            cached["index"],utc=True,)

                return self._clean_history(
                    df,
                    ticker,
                )

            except Exception as exc:
                logger.warning(
                    "Failed to restore cached yfinance history for %s: %s",
                    ticker,
                    exc,
                )

        try:
            import yfinance as yf

            ticker_object = yf.Ticker(
                ticker
            )

            df = ticker_object.history(
                period=period,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            df = self._clean_history(
                df,
                ticker,
            )

            if df is None or df.empty:
                logger.warning(
                    "yfinance returned no usable historical data for %s",
                    ticker,
                )
                return None

            set_cached_data(
                key,
                "yfinance",
                {
                    "data": (
                        df.reset_index(
                            drop=True
                        )
                        .to_dict(
                            orient="list"
                        )
                    ),
                    "index": [
                        timestamp.isoformat()
                        for timestamp in df.index
                    ],
                },
                TTL_HISTORICAL_MARKET,
            )

            return df

        except Exception as exc:
            logger.warning(
                "yfinance history fetch failed for %s: %s",
                ticker,
                exc,
            )

            return None

    @staticmethod
    def _clean_history(
        df,
        ticker: str,
    ):
        """
        Normalize and clean yfinance historical OHLCV data.

        Handles:
        - empty DataFrames
        - MultiIndex columns
        - non-numeric values
        - NaN / infinite Close values
        - duplicate dates
        - unsorted dates
        """
        if df is None or df.empty:
            return None

        import pandas as pd

        df = df.copy()

        # Defensive support for yfinance MultiIndex output.
        if isinstance(
            df.columns,
            pd.MultiIndex,
        ):
            try:
                if ticker in df.columns.get_level_values(-1):
                    df = df.xs(
                        ticker,
                        axis=1,
                        level=-1,
                    )
                elif ticker in df.columns.get_level_values(0):
                    df = df.xs(
                        ticker,
                        axis=1,
                        level=0,
                    )
                else:
                    # If only one ticker is present, flatten the first level.
                    df.columns = [
                        column[0]
                        if isinstance(column, tuple)
                        else column
                        for column in df.columns
                    ]

            except Exception as exc:
                logger.warning(
                    "Could not normalize MultiIndex history for %s: %s",
                    ticker,
                    exc,
                )

                return None

        required_column = "Close"

        if required_column not in df.columns:
            logger.warning(
                "Historical data for %s has no Close column. Columns=%s",
                ticker,
                list(df.columns),
            )

            return None

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                df.loc[
                    ~np.isfinite(df[column]),
                    column,
                ] = np.nan

        # A technical observation is unusable without a finite close.
        df = df.dropna(
            subset=["Close"]
        )

        if df.empty:
            return None

        # Remove duplicate timestamps and ensure chronological order.
        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

        df = df.sort_index()

        return df

    def get_snapshot(
        self,
        ticker: str,
    ) -> MarketDataSnapshot:
        ticker = ticker.upper().strip()

        snap = MarketDataSnapshot(
            ticker=ticker
        )

        errors: list[str] = []

        # -------------------------------------------------------------------
        # Latest quote
        # -------------------------------------------------------------------

        quote = None

        try:
            quote = finnhub_client.quote(
                ticker
            )

        except Exception as exc:
            errors.append(
                f"Finnhub quote failed: {exc}"
            )

        # -------------------------------------------------------------------
        # Historical data
        # -------------------------------------------------------------------

        # Request two years so one-year metrics and SMA200 have sufficient
        # observations even around holidays and incomplete trading periods.
        hist = self.get_history(
            ticker,
            period="2y",
        )

        if hist is None or hist.empty:
            errors.append(
                "yfinance historical data unavailable."
            )

            # Preserve a valid Finnhub quote even if history fails.
            quote_price = (
                _clean_float(
                    quote.get("c")
                )
                if quote
                else None
            )

            if quote_price is not None and quote_price > 0:
                snap.latest_price = quote_price
                snap.source = "Finnhub (quote)"
                snap.as_of = datetime.now(
                    timezone.utc
                ).isoformat()
            else:
                snap.source = "unavailable"

            snap.errors = errors

            return snap

        # -------------------------------------------------------------------
        # Clean arrays
        # -------------------------------------------------------------------

        closes = _clean_numeric_array(
            hist["Close"].to_numpy(
                dtype=float
            )
        )

        if len(closes) == 0:
            errors.append(
                "yfinance history contained no finite closing prices."
            )

            snap.errors = errors
            snap.source = "unavailable"

            return snap

        if "Volume" in hist.columns:
            volumes = _clean_numeric_array(
                hist["Volume"].to_numpy(
                    dtype=float
                )
            )
        else:
            volumes = np.array(
                [],
                dtype=float,
            )

        # -------------------------------------------------------------------
        # Latest price and provenance
        # -------------------------------------------------------------------

        quote_price = (
            _clean_float(
                quote.get("c")
            )
            if quote
            else None
        )

        if quote_price is not None and quote_price > 0:
            snap.latest_price = quote_price
            snap.source = (
                "Finnhub (latest quote) + "
                "yfinance (historical daily data)"
            )

            # This is retrieval time, not necessarily exchange trade time.
            snap.as_of = datetime.now(
                timezone.utc
            ).isoformat()

        else:
            snap.latest_price = round(
                float(closes[-1]),
                2,
            )

            snap.source = (
                "yfinance (latest available historical daily close)"
            )

            snap.as_of = hist.index[-1].isoformat()

        # -------------------------------------------------------------------
        # Returns
        # -------------------------------------------------------------------

        # Use fixed approximate trading-day horizons. If insufficient history
        # exists, the metric remains unavailable.
        snap.return_1m = _pct_return(
            closes,
            21,
        )

        snap.return_3m = _pct_return(
            closes,
            63,
        )

        snap.return_6m = _pct_return(
            closes,
            126,
        )

        snap.return_1y = _pct_return(
            closes,
            252,
        )

        # -------------------------------------------------------------------
        # Moving averages
        # -------------------------------------------------------------------

        snap.sma_20 = _sma(
            closes,
            20,
        )

        snap.sma_50 = _sma(
            closes,
            50,
        )

        snap.sma_200 = _sma(
            closes,
            200,
        )

        # -------------------------------------------------------------------
        # Momentum
        # -------------------------------------------------------------------

        snap.rsi_14 = _rsi(
            closes,
            14,
        )

        (
            snap.macd,
            snap.macd_signal,
        ) = _macd(
            closes
        )

        # -------------------------------------------------------------------
        # Volatility and drawdown
        # -------------------------------------------------------------------

        # Restrict these explicitly to approximately the latest 52 trading
        # weeks rather than the full two-year retrieval window.
        one_year_closes = (
            closes[-252:]
            if len(closes) >= 252
            else closes
        )

        snap.historical_volatility_annualized = (
            _historical_volatility(
                one_year_closes
            )
        )

        snap.max_drawdown_1y = _max_drawdown(
            one_year_closes
        )

        # -------------------------------------------------------------------
        # Volume
        # -------------------------------------------------------------------

        if len(volumes) >= 1:
            volume_window = volumes[-30:]

            avg_volume = np.mean(
                volume_window
            )

            snap.avg_volume_30d = (
                round(
                    float(avg_volume),
                    0,
                )
                if np.isfinite(avg_volume)
                else None
            )
        else:
            snap.avg_volume_30d = None

        # -------------------------------------------------------------------
        # 52-week range
        # -------------------------------------------------------------------

        week52_window = (
            closes[-252:]
            if len(closes) >= 252
            else closes
        )

        if len(week52_window) > 0:
            high = np.max(
                week52_window
            )

            low = np.min(
                week52_window
            )

            snap.week52_high = (
                round(
                    float(high),
                    2,
                )
                if np.isfinite(high)
                else None
            )

            snap.week52_low = (
                round(
                    float(low),
                    2,
                )
                if np.isfinite(low)
                else None
            )

        snap.errors = errors

        return snap


market_client = MarketDataClient()
