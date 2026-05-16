"""Backtest tools: SIP, dip-SIP, best SIP day."""

import numpy as np

from mcp_nifty.helpers import get_indices, calc_xirr
from mcp_nifty.models import (
    SIPInput,
    DipSIPInput,
    IndexNameInput,
    SIPResponse,
    DipSIPResponse,
    BestSIPDayResponse,
    BestSIPDayEntry,
    ErrorResponse,
)


def backtest_sip(params: SIPInput) -> dict:
    """Backtest a fixed-day monthly SIP on an index."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"]
    dates = df["date"].values.astype("datetime64[ns]")
    closes = df["close"].values.astype(np.float64)
    ym = (df["date"].dt.year * 100 + df["date"].dt.month).values.astype(np.int32)

    unique_months = np.unique(ym)
    buy_indices = []
    for m in unique_months:
        mask = ym == m
        month_idx = np.where(mask)[0]
        month_days = dates[month_idx].astype("datetime64[D]").astype(object)
        day_nums = np.array([d.day for d in month_days])
        eligible = month_idx[day_nums >= min(params.sip_day, day_nums.max())]
        if len(eligible) == 0:
            eligible = month_idx[-1:]
        buy_indices.append(eligible[0])

    buy_indices = np.array(buy_indices)
    buy_closes = closes[buy_indices]
    buy_dates = dates[buy_indices]

    n = len(buy_indices)
    units_per = params.monthly_amount / buy_closes
    total_units = float(np.sum(units_per))
    total_invested = params.monthly_amount * n

    final_price = closes[-1]
    final_value = total_units * final_price

    cf_dates = np.append(buy_dates, dates[-1])
    cf_amounts = np.append(np.full(n, -params.monthly_amount, dtype=np.float64), final_value)
    xirr = calc_xirr(cf_dates, cf_amounts)

    years = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(np.float64) / 365.25

    return SIPResponse(
        index_name=params.index_name,
        sip_day=params.sip_day,
        monthly_amount=params.monthly_amount,
        total_invested=round(total_invested, 2),
        final_value=round(final_value, 2),
        wealth_multiple=round(final_value / total_invested, 2) if total_invested > 0 else 0,
        xirr_pct=round(xirr, 2),
        years=round(float(years), 1),
        num_installments=n,
    ).model_dump()


def backtest_dip_sip(params: DipSIPInput) -> dict:
    """Backtest a dip-based SIP strategy."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"]
    dates = df["date"].values.astype("datetime64[ns]")
    closes = df["close"].values.astype(np.float64)
    returns = df["daily_return"].values.astype(np.float64)
    ym = (df["date"].dt.year * 100 + df["date"].dt.month).values.astype(np.int32)
    n = len(dates)

    cash = 0.0
    total_contributed = 0.0
    total_units = 0.0
    num_deployments = 0

    month_starts = np.zeros(n, dtype=bool)
    month_starts[0] = True
    month_starts[1:] = ym[1:] != ym[:-1]

    cf_dates_list = []
    cf_amounts_list = []

    for i in range(n):
        if month_starts[i]:
            cash += params.monthly_amount
            total_contributed += params.monthly_amount
            cf_dates_list.append(dates[i])
            cf_amounts_list.append(-params.monthly_amount)

        if returns[i] <= -params.dip_threshold and cash > 0:
            units = cash / closes[i]
            total_units += units
            num_deployments += 1
            cash = 0.0

    final_price = closes[-1]
    final_value = total_units * final_price + cash
    years = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(np.float64) / 365.25

    cf_dates_list.append(dates[-1])
    cf_amounts_list.append(final_value)
    xirr = calc_xirr(
        np.array(cf_dates_list, dtype="datetime64[ns]"),
        np.array(cf_amounts_list, dtype=np.float64),
    )

    return DipSIPResponse(
        index_name=params.index_name,
        dip_threshold_pct=params.dip_threshold,
        monthly_amount=params.monthly_amount,
        total_contributed=round(total_contributed, 2),
        cash_deployed=round(total_contributed - cash, 2),
        cash_remaining=round(cash, 2),
        final_value=round(final_value, 2),
        wealth_multiple=round(final_value / total_contributed, 2) if total_contributed > 0 else 0,
        xirr_pct=round(xirr, 2),
        years=round(float(years), 1),
        num_deployments=num_deployments,
    ).model_dump()


def find_best_sip_day(params: IndexNameInput) -> dict:
    """Find optimal SIP day of month by backtesting all days 1-28."""
    indices = get_indices()
    if params.index_name not in indices:
        return ErrorResponse(error=f"Index '{params.index_name}' not found.").model_dump()

    df = indices[params.index_name]["df"]
    dates = df["date"].values.astype("datetime64[ns]")
    closes = df["close"].values.astype(np.float64)
    ym = (df["date"].dt.year * 100 + df["date"].dt.month).values.astype(np.int32)
    unique_months = np.unique(ym)
    monthly_amount = 10000.0

    day_results = []
    for day in range(1, 29):
        buy_indices = []
        for m in unique_months:
            mask = ym == m
            month_idx = np.where(mask)[0]
            month_days = dates[month_idx].astype("datetime64[D]").astype(object)
            day_nums = np.array([d.day for d in month_days])
            eligible = month_idx[day_nums >= min(day, day_nums.max())]
            if len(eligible) == 0:
                eligible = month_idx[-1:]
            buy_indices.append(eligible[0])

        buy_indices_arr = np.array(buy_indices)
        buy_closes = closes[buy_indices_arr]
        buy_dates = dates[buy_indices_arr]
        n = len(buy_indices_arr)

        units_per = monthly_amount / buy_closes
        total_units = float(np.sum(units_per))
        final_value = total_units * closes[-1]

        cf_dates = np.append(buy_dates, dates[-1])
        cf_amounts = np.append(np.full(n, -monthly_amount, dtype=np.float64), final_value)
        xirr = calc_xirr(cf_dates, cf_amounts)

        day_results.append(BestSIPDayEntry(sip_day=day, xirr_pct=round(xirr, 2)))

    best = max(day_results, key=lambda x: x.xirr_pct)
    worst = min(day_results, key=lambda x: x.xirr_pct)

    return BestSIPDayResponse(
        index_name=params.index_name,
        best_day=best.sip_day,
        best_xirr_pct=best.xirr_pct,
        worst_day=worst.sip_day,
        worst_xirr_pct=worst.xirr_pct,
        spread_pct=round(best.xirr_pct - worst.xirr_pct, 2),
        all_days=day_results,
    ).model_dump()
