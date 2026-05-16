"""Analysis tools: drop stats, CAGR, compare indices, top movers."""

import pandas as pd

from mcp_nifty.helpers import get_indices, load_top_movers
from mcp_nifty.models import (
    DropStatsInput,
    CAGRInput,
    CompareInput,
    TopMoversInput,
    DropStatsResponse,
    DropStatEntry,
    CAGRResponse,
    CompareEntry,
    TopMoversResponse,
    WindowMovers,
    TopMoversEntry,
    ErrorResponse,
)


def get_drop_stats(params: DropStatsInput) -> dict:
    """Analyze days with significant drops for an index."""
    thresholds = params.thresholds or [1.0, 1.5, 2.0, 3.0, 5.0]

    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"]
    total_days = len(df)

    stats = []
    for t in thresholds:
        mask = df["daily_return"] <= -t
        count = int(mask.sum())
        pct = round(count / total_days * 100, 2) if total_days > 0 else 0
        avg_drop = round(float(df.loc[mask, "daily_return"].mean()), 2) if count > 0 else None
        worst = round(float(df.loc[mask, "daily_return"].min()), 2) if count > 0 else None
        stats.append(DropStatEntry(
            threshold_pct=t,
            days_count=count,
            pct_of_trading_days=pct,
            avg_drop_pct=avg_drop,
            worst_drop_pct=worst,
        ))

    return DropStatsResponse(
        index_name=params.index_name,
        total_trading_days=total_days,
        drop_stats=stats,
    ).model_dump()


def get_cagr(params: CAGRInput) -> dict:
    """Compute CAGR for an index between two dates."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"]
    start_dt = pd.to_datetime(params.start_date, format="%d-%b-%Y")
    end_dt = pd.to_datetime(params.end_date, format="%d-%b-%Y") if params.end_date else df["date"].max()

    start_row = df[df["date"] >= start_dt].head(1)
    end_row = df[df["date"] <= end_dt].tail(1)

    if start_row.empty or end_row.empty:
        return ErrorResponse(error="No data available for the specified date range.").model_dump()

    start_price = float(start_row["close"].iloc[0])
    end_price = float(end_row["close"].iloc[0])
    actual_start = start_row["date"].iloc[0]
    actual_end = end_row["date"].iloc[0]
    years = (actual_end - actual_start).days / 365.25

    if years <= 0 or start_price <= 0:
        return ErrorResponse(error="Invalid date range or price data.").model_dump()

    cagr = ((end_price / start_price) ** (1.0 / years) - 1.0) * 100

    return CAGRResponse(
        index_name=params.index_name,
        start_date=actual_start.strftime("%d-%b-%Y"),
        end_date=actual_end.strftime("%d-%b-%Y"),
        start_price=round(start_price, 2),
        end_price=round(end_price, 2),
        years=round(years, 2),
        cagr_pct=round(cagr, 2),
        total_return_pct=round((end_price / start_price - 1) * 100, 2),
    ).model_dump()


def compare_indices(params: CompareInput) -> list[dict]:
    """Compare multiple indices on a given metric."""
    indices = get_indices()
    results = []

    for name in params.index_names:
        if name not in indices:
            results.append(CompareEntry(index_name=name, error="not found").model_dump())
            continue

        df = indices[name]["df"]
        if params.start_date:
            df = df[df["date"] >= pd.to_datetime(params.start_date, format="%d-%b-%Y")]

        if len(df) < 2:
            results.append(CompareEntry(index_name=name, error="insufficient data").model_dump())
            continue

        start_price = float(df["close"].iloc[0])
        end_price = float(df["close"].iloc[-1])
        years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        cagr = ((end_price / start_price) ** (1.0 / years) - 1.0) * 100 if years > 0 else 0

        results.append(CompareEntry(
            index_name=name,
            start_date=df["date"].iloc[0].strftime("%d-%b-%Y"),
            end_date=df["date"].iloc[-1].strftime("%d-%b-%Y"),
            cagr_pct=round(cagr, 2),
            volatility_pct=round(float(df["daily_return"].std()), 4),
            worst_drop_pct=round(float(df["daily_return"].min()), 2),
            trading_days=len(df),
        ).model_dump())

    sort_key = {
        "cagr": "cagr_pct",
        "volatility": "volatility_pct",
        "worst_drop": "worst_drop_pct",
    }.get(params.metric, "cagr_pct")

    valid = [r for r in results if r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]
    valid.sort(key=lambda x: x.get(sort_key, 0), reverse=(params.metric != "volatility"))

    return valid + errors


def get_top_movers(params: TopMoversInput) -> dict:
    """Get top gainers and losers across time windows."""
    data = load_top_movers()
    if not data:
        return ErrorResponse(error="top_movers.json not found. Run trend.py first.").model_dump()

    windows_raw = data.get("windows", {})

    if params.window:
        if params.window not in windows_raw:
            return ErrorResponse(
                error=f"Window '{params.window}' not found. Available: {list(windows_raw.keys())}"
            ).model_dump()
        windows_raw = {params.window: windows_raw[params.window]}

    windows = {}
    for key, val in windows_raw.items():
        windows[key] = WindowMovers(
            top_gainers=[TopMoversEntry(**e) for e in val.get("top_gainers", [])],
            top_losers=[TopMoversEntry(**e) for e in val.get("top_losers", [])],
        )

    return TopMoversResponse(
        requested_end_date=data.get("requested_end_date"),
        generated_at=data.get("generated_at"),
        windows=windows,
    ).model_dump()
