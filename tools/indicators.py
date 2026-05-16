"""Technical indicator and strategy tools."""

import numpy as np
import pandas as pd

from mcp_nifty.helpers import get_indices
from mcp_nifty.models import (
    TechnicalIndicatorsInput,
    MovingAverageSignalsInput,
    IndexNameInput,
    MomentumInput,
    VolatilityInput,
    MeanReversionInput,
    TechnicalRecord,
    TechnicalIndicatorsResponse,
    CrossoverSignal,
    MovingAverageSignalsResponse,
    DrawdownEpisode,
    DrawdownResponse,
    MomentumEntry,
    MomentumRankingResponse,
    VolatilityStats,
    VolDailyRecord,
    VolatilityResponse,
    MeanReversionEvent,
    MeanReversionResponse,
    ErrorResponse,
)


def get_technical_indicators(params: TechnicalIndicatorsInput) -> dict:
    """Compute SMA, EMA, RSI, Bollinger Bands, MACD for an index."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"].copy()
    close = df["close"]

    # SMA
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    # EMA
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-period, 2 std)
    df["bb_middle"] = df["sma_20"]
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std

    # MACD
    df["macd_line"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    result_df = df.tail(params.last_n_days)
    records = []
    for _, row in result_df.iterrows():
        records.append(TechnicalRecord(
            date=row["date"].strftime("%d-%b-%Y"),
            close=round(float(row["close"]), 2),
            sma_20=round(float(row["sma_20"]), 2) if pd.notna(row["sma_20"]) else None,
            sma_50=round(float(row["sma_50"]), 2) if pd.notna(row["sma_50"]) else None,
            sma_200=round(float(row["sma_200"]), 2) if pd.notna(row["sma_200"]) else None,
            rsi_14=round(float(row["rsi_14"]), 2) if pd.notna(row["rsi_14"]) else None,
            bb_upper=round(float(row["bb_upper"]), 2) if pd.notna(row["bb_upper"]) else None,
            bb_lower=round(float(row["bb_lower"]), 2) if pd.notna(row["bb_lower"]) else None,
            macd_line=round(float(row["macd_line"]), 4) if pd.notna(row["macd_line"]) else None,
            macd_signal=round(float(row["macd_signal"]), 4) if pd.notna(row["macd_signal"]) else None,
            macd_histogram=round(float(row["macd_histogram"]), 4) if pd.notna(row["macd_histogram"]) else None,
        ))

    latest = records[-1] if records else TechnicalRecord(date="", close=0)

    return TechnicalIndicatorsResponse(
        index_name=params.index_name,
        latest_snapshot=latest,
        data=records,
    ).model_dump()


def get_moving_average_signals(params: MovingAverageSignalsInput) -> dict:
    """Detect golden/death cross signals."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"].copy()
    close = df["close"]
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df = df.dropna(subset=["sma_50", "sma_200"])

    df_window = df.tail(params.lookback_days).copy()
    df_window["above"] = df_window["sma_50"] > df_window["sma_200"]
    df_window["cross"] = df_window["above"].astype(int).diff()

    signals = []
    for _, row in df_window[df_window["cross"] != 0].iterrows():
        if pd.isna(row["cross"]):
            continue
        signals.append(CrossoverSignal(
            date=row["date"].strftime("%d-%b-%Y"),
            type="golden_cross" if row["cross"] > 0 else "death_cross",
            close=round(float(row["close"]), 2),
            sma_50=round(float(row["sma_50"]), 2),
            sma_200=round(float(row["sma_200"]), 2),
        ))

    latest = df.iloc[-1]
    current_trend = "bullish" if latest["sma_50"] > latest["sma_200"] else "bearish"
    price_vs_sma200 = "above" if latest["close"] > latest["sma_200"] else "below"

    return MovingAverageSignalsResponse(
        index_name=params.index_name,
        current_trend=current_trend,
        price_vs_sma200=price_vs_sma200,
        latest_close=round(float(latest["close"]), 2),
        latest_sma_50=round(float(latest["sma_50"]), 2),
        latest_sma_200=round(float(latest["sma_200"]), 2),
        signals=signals,
    ).model_dump()


def get_drawdown_analysis(params: IndexNameInput) -> dict:
    """Analyze drawdowns: max drawdown, current drawdown, top episodes."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"].copy()
    closes = df["close"].values.astype(np.float64)
    dates = df["date"].values

    running_max = np.maximum.accumulate(closes)
    drawdowns = (closes - running_max) / running_max * 100

    max_dd_idx = np.argmin(drawdowns)
    max_dd_pct = float(drawdowns[max_dd_idx])
    peak_idx = np.argmax(closes[:max_dd_idx + 1])
    current_dd = float(drawdowns[-1])
    ath_idx = np.argmax(closes)

    # Find drawdown episodes
    episodes = []
    in_dd = False
    dd_start = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] < -5 and not in_dd:
            in_dd = True
            dd_start = np.argmax(closes[:i + 1])
        elif drawdowns[i] >= 0 and in_dd:
            trough_idx = dd_start + np.argmin(drawdowns[dd_start:i])
            episodes.append(DrawdownEpisode(
                peak_date=pd.Timestamp(dates[dd_start]).strftime("%d-%b-%Y"),
                trough_date=pd.Timestamp(dates[trough_idx]).strftime("%d-%b-%Y"),
                recovery_date=pd.Timestamp(dates[i]).strftime("%d-%b-%Y"),
                drawdown_pct=round(float(drawdowns[trough_idx]), 2),
                days_to_trough=int(trough_idx - dd_start),
                days_to_recover=int(i - trough_idx),
            ))
            in_dd = False

    if in_dd:
        trough_idx = dd_start + np.argmin(drawdowns[dd_start:])
        episodes.append(DrawdownEpisode(
            peak_date=pd.Timestamp(dates[dd_start]).strftime("%d-%b-%Y"),
            trough_date=pd.Timestamp(dates[trough_idx]).strftime("%d-%b-%Y"),
            recovery_date=None,
            drawdown_pct=round(float(drawdowns[trough_idx]), 2),
            days_to_trough=int(trough_idx - dd_start),
            days_to_recover=None,
        ))

    episodes.sort(key=lambda x: x.drawdown_pct)

    return DrawdownResponse(
        index_name=params.index_name,
        max_drawdown_pct=round(max_dd_pct, 2),
        max_drawdown_peak_date=pd.Timestamp(dates[peak_idx]).strftime("%d-%b-%Y"),
        max_drawdown_trough_date=pd.Timestamp(dates[max_dd_idx]).strftime("%d-%b-%Y"),
        current_drawdown_pct=round(current_dd, 2),
        all_time_high=round(float(closes[ath_idx]), 2),
        ath_date=pd.Timestamp(dates[ath_idx]).strftime("%d-%b-%Y"),
        top_drawdown_episodes=episodes[:5],
    ).model_dump()


def get_momentum_ranking(params: MomentumInput) -> dict:
    """Rank indices by momentum over a specified window."""
    window_days = {
        "1_month": 22,
        "3_months": 66,
        "6_months": 132,
        "1_year": 252,
    }
    days = window_days.get(params.window, 132)

    indices = get_indices()
    rankings = []

    for name, info in indices.items():
        df = info["df"]
        if len(df) < days + 1:
            continue
        current_price = float(df["close"].iloc[-1])
        past_price = float(df["close"].iloc[-(days + 1)])
        if past_price <= 0:
            continue
        momentum = (current_price / past_price - 1) * 100
        rankings.append(MomentumEntry(
            index_name=name,
            index_type=info["type"],
            momentum_pct=round(momentum, 2),
            current_price=round(current_price, 2),
            price_n_days_ago=round(past_price, 2),
        ))

    rankings.sort(key=lambda x: x.momentum_pct, reverse=True)

    return MomentumRankingResponse(
        window=params.window,
        lookback_days=days,
        top_gainers=rankings[:params.top_n],
        top_losers=rankings[-params.top_n:][::-1] if len(rankings) >= params.top_n else rankings[::-1],
        total_indices_ranked=len(rankings),
    ).model_dump()


def get_volatility_analysis(params: VolatilityInput) -> dict:
    """Analyze rolling volatility and detect regimes."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"].copy()
    returns = df["daily_return"] / 100

    rolling_vol = returns.rolling(params.rolling_window).std() * np.sqrt(252) * 100
    df["rolling_vol"] = rolling_vol

    current_vol = float(rolling_vol.iloc[-1])
    vol_history = rolling_vol.dropna()

    percentile = float((vol_history < current_vol).sum() / len(vol_history) * 100)

    if percentile > 80:
        regime = "high_volatility"
    elif percentile < 20:
        regime = "low_volatility"
    else:
        regime = "normal"

    recent_vol = float(vol_history.tail(20).mean())
    prev_vol = float(vol_history.iloc[-40:-20].mean()) if len(vol_history) > 40 else recent_vol
    vol_trend = "increasing" if recent_vol > prev_vol * 1.1 else ("decreasing" if recent_vol < prev_vol * 0.9 else "stable")

    recent_data = []
    for i in range(-min(20, len(df)), 0):
        if pd.notna(rolling_vol.iloc[i]):
            recent_data.append(VolDailyRecord(
                date=df["date"].iloc[i].strftime("%d-%b-%Y"),
                vol_pct=round(float(rolling_vol.iloc[i]), 2),
            ))

    return VolatilityResponse(
        index_name=params.index_name,
        current_annualized_vol_pct=round(current_vol, 2),
        vol_percentile=round(percentile, 1),
        regime=regime,
        vol_trend=vol_trend,
        vol_stats=VolatilityStats(
            mean_pct=round(float(vol_history.mean()), 2),
            median_pct=round(float(vol_history.median()), 2),
            min_pct=round(float(vol_history.min()), 2),
            max_pct=round(float(vol_history.max()), 2),
            current_vs_mean=round(current_vol / float(vol_history.mean()), 2),
        ),
        recent_daily_vol_data=recent_data,
    ).model_dump()


def get_mean_reversion_signals(params: MeanReversionInput) -> dict:
    """Identify overbought/oversold conditions via z-score."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"].copy()
    close = df["close"]

    sma = close.rolling(params.lookback).mean()
    std = close.rolling(params.lookback).std()
    df["z_score"] = (close - sma) / std.replace(0, np.nan)

    df = df.dropna(subset=["z_score"])
    current_z = float(df["z_score"].iloc[-1])

    if current_z >= params.z_threshold:
        signal = "overbought"
    elif current_z <= -params.z_threshold:
        signal = "oversold"
    else:
        signal = "neutral"

    # Recent signal events
    recent = df.tail(252)
    events = []
    prev_signal = "neutral"
    for _, row in recent.iterrows():
        z = float(row["z_score"])
        if z >= params.z_threshold and prev_signal != "overbought":
            events.append(MeanReversionEvent(
                date=row["date"].strftime("%d-%b-%Y"),
                type="overbought",
                z_score=round(z, 2),
                close=round(float(row["close"]), 2),
            ))
            prev_signal = "overbought"
        elif z <= -params.z_threshold and prev_signal != "oversold":
            events.append(MeanReversionEvent(
                date=row["date"].strftime("%d-%b-%Y"),
                type="oversold",
                z_score=round(z, 2),
                close=round(float(row["close"]), 2),
            ))
            prev_signal = "oversold"
        elif -params.z_threshold < z < params.z_threshold:
            prev_signal = "neutral"

    return MeanReversionResponse(
        index_name=params.index_name,
        lookback_days=params.lookback,
        z_threshold=params.z_threshold,
        current_z_score=round(current_z, 2),
        current_signal=signal,
        current_close=round(float(df["close"].iloc[-1]), 2),
        sma_value=round(float(sma.iloc[-1]), 2),
        distance_from_mean_pct=round((float(close.iloc[-1]) / float(sma.iloc[-1]) - 1) * 100, 2),
        recent_signals=events,
    ).model_dump()
