"""Data retrieval tools: list indices, get data, fetch live prices."""

from datetime import datetime, timedelta

import pandas as pd

from mcp_nifty.helpers import get_indices, load_summary, fetch_range, date_to_str
from mcp_nifty.models import (
    ListIndicesInput,
    IndexNameInput,
    DateRangeInput,
    FetchLivePriceInput,
    IndexSummaryResponse,
    IndexDataRecord,
    IndexListEntry,
    LivePriceResponse,
    ErrorResponse,
)


def list_indices(params: ListIndicesInput) -> list[dict]:
    """List all available Nifty indices with metadata."""
    summary = load_summary()
    if not summary:
        indices = get_indices()
        summary = [
            {
                "INDEX_NAME": name,
                "INDEX_TYPE": info["type"],
                "FIRST_DATE": info["first"],
                "LAST_DATE": info["last"],
                "TOTAL_DATA_POINTS": info["points"],
            }
            for name, info in indices.items()
        ]

    if params.index_type:
        summary = [s for s in summary if params.index_type.lower() in s.get("INDEX_TYPE", "").lower()]

    return [IndexListEntry(**s).model_dump() for s in summary]


def get_index_summary(params: IndexNameInput) -> dict:
    """Get summary information for a specific index."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    info = indices[params.index_name]
    df = info["df"]
    return IndexSummaryResponse(
        index_name=params.index_name,
        index_type=info["type"],
        first_date=info["first"],
        last_date=info["last"],
        total_trading_days=len(df),
        first_close=round(float(df["close"].iloc[0]), 2),
        last_close=round(float(df["close"].iloc[-1]), 2),
        avg_daily_return_pct=round(float(df["daily_return"].mean()), 4),
        std_daily_return_pct=round(float(df["daily_return"].std()), 4),
        worst_single_day_pct=round(float(df["daily_return"].min()), 2),
        best_single_day_pct=round(float(df["daily_return"].max()), 2),
    ).model_dump()


def get_index_data(params: DateRangeInput) -> list[dict]:
    """Get historical price data for a specific index."""
    indices = get_indices()
    if params.index_name not in indices:
        return [ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()]

    df = indices[params.index_name]["df"].copy()

    if params.last_n_days:
        df = df.tail(params.last_n_days)
    else:
        if params.start_date:
            df = df[df["date"] >= pd.to_datetime(params.start_date, format="%d-%b-%Y")]
        if params.end_date:
            df = df[df["date"] <= pd.to_datetime(params.end_date, format="%d-%b-%Y")]

    if len(df) > 500:
        df = df.tail(500)

    return [
        IndexDataRecord(
            date=row["date"].strftime("%d-%b-%Y"),
            close=round(float(row["close"]), 2),
            daily_return_pct=round(float(row["daily_return"]), 4),
        ).model_dump()
        for _, row in df.iterrows()
    ]


def fetch_live_price(params: FetchLivePriceInput) -> dict:
    """Fetch the latest available price for an index from the NSE API."""
    if params.date:
        end_dt = datetime.strptime(params.date, "%d-%b-%Y")
    else:
        end_dt = datetime.now()

    start_dt = end_dt - timedelta(days=7)
    data = fetch_range(params.index_name, date_to_str(start_dt), date_to_str(end_dt))

    if not data:
        return ErrorResponse(error=f"No data available for '{params.index_name}' near {date_to_str(end_dt)}").model_dump()

    for row in reversed(data):
        ds = (
            row.get("HistoricalDate")
            or row.get("Date")
            or row.get("DATE")
            or row.get("TradeDate")
        )
        val = (
            row.get("CLOSE")
            or row.get("Close")
            or row.get("Index Value")
            or row.get("INDEX_VALUE")
        )
        if ds and val is not None:
            try:
                return LivePriceResponse(
                    index_name=params.index_name,
                    date=ds,
                    close=float(val),
                ).model_dump()
            except (TypeError, ValueError):
                continue

    return ErrorResponse(error="Could not parse API response.").model_dump()
