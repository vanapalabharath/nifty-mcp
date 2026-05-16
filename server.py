"""FastMCP server entry point — registers all tools and runs the server."""

from mcp.server.fastmcp import FastMCP

from mcp_nifty.models import (
    ListIndicesInput,
    IndexNameInput,
    DateRangeInput,
    FetchLivePriceInput,
    DropStatsInput,
    CAGRInput,
    CompareInput,
    TopMoversInput,
    SIPInput,
    DipSIPInput,
    TechnicalIndicatorsInput,
    MovingAverageSignalsInput,
    MomentumInput,
    VolatilityInput,
    MeanReversionInput,
)
from mcp_nifty.tools.data import (
    list_indices as _list_indices,
    get_index_summary as _get_index_summary,
    get_index_data as _get_index_data,
    fetch_live_price as _fetch_live_price,
)
from mcp_nifty.tools.analysis import (
    get_drop_stats as _get_drop_stats,
    get_cagr as _get_cagr,
    compare_indices as _compare_indices,
    get_top_movers as _get_top_movers,
)
from mcp_nifty.tools.backtest import (
    backtest_sip as _backtest_sip,
    backtest_dip_sip as _backtest_dip_sip,
    find_best_sip_day as _find_best_sip_day,
)
from mcp_nifty.tools.indicators import (
    get_technical_indicators as _get_technical_indicators,
    get_moving_average_signals as _get_moving_average_signals,
    get_drawdown_analysis as _get_drawdown_analysis,
    get_momentum_ranking as _get_momentum_ranking,
    get_volatility_analysis as _get_volatility_analysis,
    get_mean_reversion_signals as _get_mean_reversion_signals,
)

# ─── Server Instance ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "Nifty Index Analysis",
    dependencies=["numpy", "pandas", "scipy", "requests", "pydantic"],
)

# ─── Data Tools ──────────────────────────────────────────────────────────────


@mcp.tool()
def list_indices(index_type: str | None = None) -> list[dict]:
    """
    List all available Nifty indices with their metadata.

    Returns the full catalogue of indices including name, type classification,
    first/last available date, and total number of data points. Use the optional
    index_type filter to narrow results to a specific category.

    Args:
        index_type: Filter by category (e.g., 'Broad Market Indices', 'Sectoral Indices',
                    'Thematic Indices', 'Strategy Indices'). Case-insensitive partial match.

    Returns:
        List of dicts with INDEX_NAME, INDEX_TYPE, FIRST_DATE, LAST_DATE, TOTAL_DATA_POINTS.
    """
    return _list_indices(ListIndicesInput(index_type=index_type))


@mcp.tool()
def get_index_summary(index_name: str) -> dict:
    """
    Get a comprehensive statistical summary for a single index.

    Provides key metrics at a glance: historical date range, first and last closing prices,
    total trading days, average daily return, daily return standard deviation, and the
    single best and worst trading days ever recorded.

    Args:
        index_name: Exact index name (e.g., 'Nifty 50', 'Nifty Midcap 150', 'Nifty IT').

    Returns:
        Dict with index_name, index_type, first_date, last_date, total_trading_days,
        first_close, last_close, avg_daily_return_pct, std_daily_return_pct,
        worst_single_day_pct, best_single_day_pct.
    """
    return _get_index_summary(IndexNameInput(index_name=index_name))


@mcp.tool()
def get_index_data(
    index_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    last_n_days: int | None = None,
) -> list[dict]:
    """
    Retrieve historical daily closing prices and returns for an index.

    Supports flexible date filtering: use last_n_days for the most recent N trading days,
    or provide start_date/end_date for a custom range. If the result exceeds 500 rows,
    only the most recent 500 are returned to keep response sizes manageable.

    Args:
        index_name: Exact index name (e.g., 'Nifty 50').
        start_date: Start of date range in DD-Mon-YYYY format (e.g., '01-Jan-2020').
        end_date: End of date range in DD-Mon-YYYY format.
        last_n_days: If provided, returns only the last N trading days (overrides date range).

    Returns:
        List of dicts with date, close, daily_return_pct for each trading day.
    """
    return _get_index_data(DateRangeInput(
        index_name=index_name,
        start_date=start_date,
        end_date=end_date,
        last_n_days=last_n_days,
    ))


@mcp.tool()
def fetch_live_price(index_name: str, date: str | None = None) -> dict:
    """
    Fetch the most recent closing price for an index directly from the NSE Nifty Indices API.

    Looks back up to 7 days from the target date to find the latest available trading day.
    Useful for getting near-real-time prices that may not yet be in the cached dataset.

    Args:
        index_name: Exact index name (e.g., 'Nifty 50').
        date: Target date in DD-Mon-YYYY format. Defaults to today if not provided.

    Returns:
        Dict with index_name, date (actual trading date found), and close price.
    """
    return _fetch_live_price(FetchLivePriceInput(index_name=index_name, date=date))


# ─── Analysis Tools ──────────────────────────────────────────────────────────


@mcp.tool()
def get_drop_stats(index_name: str, thresholds: list[float] | None = None) -> dict:
    """
    Analyze the frequency and severity of market drops for an index.

    For each drop threshold, calculates how many trading days saw a decline equal to or
    greater than that threshold, what percentage of total trading days that represents,
    the average drop on those days, and the single worst drop observed.

    Useful for understanding tail risk and calibrating dip-buying strategies.

    Args:
        index_name: Exact index name.
        thresholds: List of drop thresholds in percent (default: [1.0, 1.5, 2.0, 3.0, 5.0]).
                    E.g., threshold=2.0 means "days where close fell >= 2% from previous close".

    Returns:
        Dict with index_name, total_trading_days, and drop_stats (list of per-threshold results).
    """
    return _get_drop_stats(DropStatsInput(index_name=index_name, thresholds=thresholds))


@mcp.tool()
def get_cagr(index_name: str, start_date: str, end_date: str | None = None) -> dict:
    """
    Compute the Compound Annual Growth Rate (CAGR) for an index over a date range.

    Finds the nearest available trading days to the specified start and end dates,
    then calculates the annualized return assuming compounding. Also returns the
    absolute total return percentage.

    Args:
        index_name: Exact index name.
        start_date: Start date in DD-Mon-YYYY format (e.g., '01-Jan-2015').
        end_date: End date in DD-Mon-YYYY format. Defaults to the latest available date.

    Returns:
        Dict with index_name, actual start/end dates used, start/end prices,
        years, cagr_pct, and total_return_pct.
    """
    return _get_cagr(CAGRInput(index_name=index_name, start_date=start_date, end_date=end_date))


@mcp.tool()
def compare_indices(
    index_names: list[str],
    start_date: str | None = None,
    metric: str = "cagr",
) -> list[dict]:
    """
    Compare multiple indices side-by-side on key performance and risk metrics.

    Computes CAGR, daily volatility (std dev of returns), and worst single-day drop
    for each index from a common start date. Results are sorted by the chosen metric.

    Ideal for deciding between investment options (e.g., Nifty 50 vs Midcap 150 vs IT).

    Args:
        index_names: List of index names to compare (e.g., ['Nifty 50', 'Nifty Midcap 150']).
        start_date: Common start date in DD-Mon-YYYY format. If None, uses each index's full history.
        metric: Sort metric — 'cagr' (highest first), 'volatility' (lowest first), 'worst_drop'.

    Returns:
        List of dicts sorted by metric, each with index_name, start_date, end_date,
        cagr_pct, volatility_pct, worst_drop_pct, trading_days.
    """
    return _compare_indices(CompareInput(index_names=index_names, start_date=start_date, metric=metric))


@mcp.tool()
def get_top_movers(window: str | None = None) -> dict:
    """
    Get the top 5 gaining and losing indices across multiple time windows.

    Uses pre-computed data (from trend.py) showing which indices moved the most
    over various periods. Helpful for identifying sector rotation and momentum shifts.

    Args:
        window: Specific time window to retrieve. Options: '1_day', '1_week', '1_month',
                '3_months', '6_months', '1_year'. If None, returns all windows.

    Returns:
        Dict with requested_end_date, generated_at timestamp, and windows mapping
        each to top_gainers and top_losers (with INDEX_NAME, prices, CHANGE_PCT).
    """
    return _get_top_movers(TopMoversInput(window=window))


# ─── Backtest Tools ──────────────────────────────────────────────────────────


@mcp.tool()
def backtest_sip(index_name: str, sip_day: int = 1, monthly_amount: float = 10000) -> dict:
    """
    Backtest a Systematic Investment Plan (SIP) with a fixed monthly investment day.

    Simulates investing a fixed amount on a specific calendar day each month over the
    entire available history of the index. On non-trading days, investment happens on the
    next available trading day. Calculates XIRR (true annualized return accounting for
    cashflow timing) and wealth multiple.

    Args:
        index_name: Exact index name.
        sip_day: Day of month to invest (1-28). E.g., 5 means invest on the 5th or next trading day.
        monthly_amount: Amount invested each month in INR (default: ₹10,000).

    Returns:
        Dict with index_name, sip_day, monthly_amount, total_invested, final_value,
        wealth_multiple, xirr_pct, years, num_installments.
    """
    return _backtest_sip(SIPInput(index_name=index_name, sip_day=sip_day, monthly_amount=monthly_amount))


@mcp.tool()
def backtest_dip_sip(index_name: str, dip_threshold: float = 2.0, monthly_amount: float = 10000) -> dict:
    """
    Backtest a dip-buying SIP strategy that deploys cash only on significant market drops.

    Each month, a fixed amount is added to a cash reserve. The entire cash reserve is
    deployed (invested) only on days when the index drops by at least the dip_threshold
    percentage from the previous close. If no dip occurs, cash accumulates until one does.

    This tests whether "buying the dip" outperforms regular SIP investing.

    Args:
        index_name: Exact index name.
        dip_threshold: Minimum daily drop (%) required to trigger deployment (default: 2.0).
                       Higher values = fewer deployments but at lower prices.
        monthly_amount: Monthly cash contribution in INR (default: ₹10,000).

    Returns:
        Dict with total_contributed, cash_deployed, cash_remaining, final_value,
        wealth_multiple, xirr_pct, years, num_deployments.
    """
    return _backtest_dip_sip(DipSIPInput(
        index_name=index_name, dip_threshold=dip_threshold, monthly_amount=monthly_amount,
    ))


@mcp.tool()
def find_best_sip_day(index_name: str) -> dict:
    """
    Find the optimal day of the month for SIP investment by exhaustive backtesting.

    Runs 28 separate SIP backtests (one for each possible day 1-28) over the full
    index history and compares the resulting XIRR. Shows which calendar day historically
    produced the best and worst returns, plus the spread between them.

    Note: The spread is typically small (0.1-0.5%), confirming that consistency matters
    more than timing within the month.

    Args:
        index_name: Exact index name.

    Returns:
        Dict with best_day, best_xirr_pct, worst_day, worst_xirr_pct, spread_pct,
        and all_days (full list of day→XIRR results).
    """
    return _find_best_sip_day(IndexNameInput(index_name=index_name))


# ─── Indicator & Strategy Tools ──────────────────────────────────────────────


@mcp.tool()
def get_technical_indicators(index_name: str, last_n_days: int = 50) -> dict:
    """
    Compute standard technical indicators for recent trading days.

    Calculates the following indicators over the index's full history, then returns
    the most recent N days:
    - SMA (Simple Moving Average): 20, 50, 200 day periods
    - EMA (Exponential Moving Average): 12, 26 day spans
    - RSI (Relative Strength Index): 14-period, range 0-100 (>70 overbought, <30 oversold)
    - Bollinger Bands: 20-period middle band ± 2 standard deviations
    - MACD: 12/26 EMA crossover with 9-period signal line and histogram

    Args:
        index_name: Exact index name.
        last_n_days: Number of recent trading days to return (default: 50, max: 500).

    Returns:
        Dict with index_name, latest_snapshot (most recent day's values),
        and data (list of daily records with all indicator values).
    """
    return _get_technical_indicators(TechnicalIndicatorsInput(index_name=index_name, last_n_days=last_n_days))


@mcp.tool()
def get_moving_average_signals(index_name: str, lookback_days: int = 252) -> dict:
    """
    Detect Golden Cross and Death Cross signals based on SMA 50/200 crossovers.

    - Golden Cross: SMA-50 crosses ABOVE SMA-200 → bullish long-term trend reversal.
    - Death Cross: SMA-50 crosses BELOW SMA-200 → bearish long-term trend reversal.

    Also reports the current trend direction and whether price is above/below the 200-day SMA.

    Args:
        index_name: Exact index name.
        lookback_days: Number of recent trading days to scan for crossover events
                       (default: 252 ≈ 1 year).

    Returns:
        Dict with current_trend ('bullish'/'bearish'), price_vs_sma200 ('above'/'below'),
        latest MA values, and signals list (date, type, close, sma_50, sma_200).
    """
    return _get_moving_average_signals(MovingAverageSignalsInput(
        index_name=index_name, lookback_days=lookback_days,
    ))


@mcp.tool()
def get_drawdown_analysis(index_name: str) -> dict:
    """
    Analyze historical drawdowns — peak-to-trough declines and recovery periods.

    Computes:
    - Maximum drawdown ever recorded (worst peak-to-trough decline).
    - Current drawdown from the all-time high.
    - Top 5 worst drawdown episodes with peak date, trough date, recovery date,
      days to reach bottom, and days to recover back to the previous high.

    Useful for understanding worst-case scenarios and typical recovery timelines.

    Args:
        index_name: Exact index name.

    Returns:
        Dict with max_drawdown_pct, current_drawdown_pct, all_time_high, ath_date,
        and top_drawdown_episodes (list of episodes with full timeline details).
    """
    return _get_drawdown_analysis(IndexNameInput(index_name=index_name))


@mcp.tool()
def get_momentum_ranking(top_n: int = 10, window: str = "6_months") -> dict:
    """
    Rank all available indices by price momentum over a chosen time window.

    Momentum is measured as the simple percentage return over the lookback period.
    Returns the top N strongest and weakest indices — useful for momentum-based
    sector rotation strategies or identifying market leadership shifts.

    Args:
        top_n: Number of top gainers and losers to return (default: 10).
        window: Lookback period — '1_month' (22 days), '3_months' (66 days),
                '6_months' (132 days), '1_year' (252 days).

    Returns:
        Dict with window, lookback_days, total_indices_ranked, top_gainers, top_losers
        (each entry has index_name, index_type, momentum_pct, current/past prices).
    """
    return _get_momentum_ranking(MomentumInput(top_n=top_n, window=window))


@mcp.tool()
def get_volatility_analysis(index_name: str, rolling_window: int = 20) -> dict:
    """
    Analyze historical and current volatility to detect market regimes.

    Computes annualized rolling volatility and places the current reading in historical
    context using percentile ranking. Classifies the market into regimes:
    - high_volatility: Current vol > 80th percentile historically.
    - low_volatility: Current vol < 20th percentile historically.
    - normal: Everything in between.

    Also detects if volatility is trending up, down, or stable.

    Args:
        index_name: Exact index name.
        rolling_window: Window in trading days for volatility calculation (default: 20).

    Returns:
        Dict with current_annualized_vol_pct, vol_percentile, regime, vol_trend,
        vol_stats (mean, median, min, max, current_vs_mean), and recent daily vol data.
    """
    return _get_volatility_analysis(VolatilityInput(index_name=index_name, rolling_window=rolling_window))


@mcp.tool()
def get_mean_reversion_signals(
    index_name: str,
    z_threshold: float = 2.0,
    lookback: int = 50,
) -> dict:
    """
    Identify overbought and oversold conditions using price z-score relative to moving average.

    Computes z-score = (price - SMA) / rolling_std. When the z-score exceeds the threshold,
    the market is statistically extended:
    - z >= +threshold → OVERBOUGHT (price far above mean, potential pullback).
    - z <= -threshold → OVERSOLD (price far below mean, potential bounce).
    - Otherwise → NEUTRAL.

    Based on mean reversion theory: extreme deviations from the mean tend to revert.

    Args:
        index_name: Exact index name.
        z_threshold: Z-score magnitude for signal generation (default: 2.0 = 2 sigma).
        lookback: Period for SMA and std dev calculation in trading days (default: 50).

    Returns:
        Dict with current_z_score, current_signal, distance_from_mean_pct,
        sma_value, and recent_signals (list of dated overbought/oversold events).
    """
    return _get_mean_reversion_signals(MeanReversionInput(
        index_name=index_name, z_threshold=z_threshold, lookback=lookback,
    ))


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
