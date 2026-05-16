"""Pydantic models for MCP tool inputs and outputs."""

from pydantic import BaseModel, Field


# ─── Input Models ────────────────────────────────────────────────────────────


class IndexNameInput(BaseModel):
    index_name: str = Field(..., description="Name of the index (e.g., 'Nifty 50', 'Nifty Midcap 150')")


class DateRangeInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    start_date: str | None = Field(None, description="Start date in DD-Mon-YYYY format (e.g., '01-Jan-2020')")
    end_date: str | None = Field(None, description="End date in DD-Mon-YYYY format")
    last_n_days: int | None = Field(None, ge=1, le=5000, description="Return only the last N trading days")


class SIPInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    sip_day: int = Field(1, ge=1, le=28, description="Day of month to invest (1-28)")
    monthly_amount: float = Field(10000, gt=0, description="Monthly SIP amount in INR")


class DipSIPInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    dip_threshold: float = Field(2.0, gt=0, le=20, description="Deploy cash when daily drop >= this %")
    monthly_amount: float = Field(10000, gt=0, description="Monthly contribution in INR")


class DropStatsInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    thresholds: list[float] | None = Field(None, description="Drop thresholds in % (default: [1.0, 1.5, 2.0, 3.0, 5.0])")


class CAGRInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    start_date: str = Field(..., description="Start date in DD-Mon-YYYY format")
    end_date: str | None = Field(None, description="End date in DD-Mon-YYYY format (defaults to latest)")


class CompareInput(BaseModel):
    index_names: list[str] = Field(..., min_length=1, description="List of index names to compare")
    start_date: str | None = Field(None, description="Common start date in DD-Mon-YYYY format")
    metric: str = Field("cagr", description="Sort metric: 'cagr', 'volatility', 'worst_drop'")


class MomentumInput(BaseModel):
    top_n: int = Field(10, ge=1, le=50, description="Number of top/bottom indices to return")
    window: str = Field("6_months", description="Time window: '1_month', '3_months', '6_months', '1_year'")


class VolatilityInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    rolling_window: int = Field(20, ge=5, le=252, description="Rolling window in trading days")


class MeanReversionInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    z_threshold: float = Field(2.0, gt=0, le=5, description="Z-score threshold for signal generation")
    lookback: int = Field(50, ge=10, le=252, description="Lookback period for mean/std calculation")


class TechnicalIndicatorsInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    last_n_days: int = Field(50, ge=1, le=500, description="Number of recent trading days to return")


class MovingAverageSignalsInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    lookback_days: int = Field(252, ge=50, le=2520, description="Trading days to scan for signals")


class FetchLivePriceInput(BaseModel):
    index_name: str = Field(..., description="Name of the index")
    date: str | None = Field(None, description="Specific date in DD-Mon-YYYY format (defaults to today)")


class ListIndicesInput(BaseModel):
    index_type: str | None = Field(None, description="Filter by index type (e.g., 'Broad Market Indices')")


class TopMoversInput(BaseModel):
    window: str | None = Field(None, description="Time window: '1_day', '1_week', '1_month', '3_months', '6_months', '1_year'")


# ─── Output Models ───────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str


class IndexSummaryResponse(BaseModel):
    index_name: str
    index_type: str
    first_date: str
    last_date: str
    total_trading_days: int
    first_close: float
    last_close: float
    avg_daily_return_pct: float
    std_daily_return_pct: float
    worst_single_day_pct: float
    best_single_day_pct: float


class IndexDataRecord(BaseModel):
    date: str
    close: float
    daily_return_pct: float


class DropStatEntry(BaseModel):
    threshold_pct: float
    days_count: int
    pct_of_trading_days: float
    avg_drop_pct: float | None
    worst_drop_pct: float | None


class DropStatsResponse(BaseModel):
    index_name: str
    total_trading_days: int
    drop_stats: list[DropStatEntry]


class SIPResponse(BaseModel):
    index_name: str
    sip_day: int
    monthly_amount: float
    total_invested: float
    final_value: float
    wealth_multiple: float
    xirr_pct: float
    years: float
    num_installments: int


class DipSIPResponse(BaseModel):
    index_name: str
    dip_threshold_pct: float
    monthly_amount: float
    total_contributed: float
    cash_deployed: float
    cash_remaining: float
    final_value: float
    wealth_multiple: float
    xirr_pct: float
    years: float
    num_deployments: int


class BestSIPDayEntry(BaseModel):
    sip_day: int
    xirr_pct: float


class BestSIPDayResponse(BaseModel):
    index_name: str
    best_day: int
    best_xirr_pct: float
    worst_day: int
    worst_xirr_pct: float
    spread_pct: float
    all_days: list[BestSIPDayEntry]


class CAGRResponse(BaseModel):
    index_name: str
    start_date: str
    end_date: str
    start_price: float
    end_price: float
    years: float
    cagr_pct: float
    total_return_pct: float


class CompareEntry(BaseModel):
    index_name: str
    start_date: str | None = None
    end_date: str | None = None
    cagr_pct: float | None = None
    volatility_pct: float | None = None
    worst_drop_pct: float | None = None
    trading_days: int | None = None
    error: str | None = None


class TechnicalRecord(BaseModel):
    date: str
    close: float
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None


class TechnicalIndicatorsResponse(BaseModel):
    index_name: str
    latest_snapshot: TechnicalRecord | dict
    data: list[TechnicalRecord]


class CrossoverSignal(BaseModel):
    date: str
    type: str
    close: float
    sma_50: float
    sma_200: float


class MovingAverageSignalsResponse(BaseModel):
    index_name: str
    current_trend: str
    price_vs_sma200: str
    latest_close: float
    latest_sma_50: float
    latest_sma_200: float
    signals: list[CrossoverSignal]


class DrawdownEpisode(BaseModel):
    peak_date: str
    trough_date: str
    recovery_date: str | None
    drawdown_pct: float
    days_to_trough: int
    days_to_recover: int | None


class DrawdownResponse(BaseModel):
    index_name: str
    max_drawdown_pct: float
    max_drawdown_peak_date: str
    max_drawdown_trough_date: str
    current_drawdown_pct: float
    all_time_high: float
    ath_date: str
    top_drawdown_episodes: list[DrawdownEpisode]


class MomentumEntry(BaseModel):
    index_name: str
    index_type: str
    momentum_pct: float
    current_price: float
    price_n_days_ago: float


class MomentumRankingResponse(BaseModel):
    window: str
    lookback_days: int
    top_gainers: list[MomentumEntry]
    top_losers: list[MomentumEntry]
    total_indices_ranked: int


class VolatilityStats(BaseModel):
    mean_pct: float
    median_pct: float
    min_pct: float
    max_pct: float
    current_vs_mean: float


class VolDailyRecord(BaseModel):
    date: str
    vol_pct: float


class VolatilityResponse(BaseModel):
    index_name: str
    current_annualized_vol_pct: float
    vol_percentile: float
    regime: str
    vol_trend: str
    vol_stats: VolatilityStats
    recent_daily_vol_data: list[VolDailyRecord]


class MeanReversionEvent(BaseModel):
    date: str
    type: str
    z_score: float
    close: float


class MeanReversionResponse(BaseModel):
    index_name: str
    lookback_days: int
    z_threshold: float
    current_z_score: float
    current_signal: str
    current_close: float
    sma_value: float
    distance_from_mean_pct: float
    recent_signals: list[MeanReversionEvent]


class LivePriceResponse(BaseModel):
    index_name: str
    date: str
    close: float


class IndexListEntry(BaseModel):
    INDEX_NAME: str
    INDEX_TYPE: str | None = None
    FIRST_DATE: str | None = None
    LAST_DATE: str | None = None
    TOTAL_DATA_POINTS: int | None = None


class TopMoversEntry(BaseModel):
    INDEX_NAME: str
    START_ACTUAL_DATE: str | None = None
    START_PRICE: float | None = None
    END_ACTUAL_DATE: str | None = None
    END_PRICE: float | None = None
    CHANGE_PCT: float | None = None


class WindowMovers(BaseModel):
    top_gainers: list[TopMoversEntry]
    top_losers: list[TopMoversEntry]


class TopMoversResponse(BaseModel):
    requested_end_date: str | None = None
    generated_at: str | None = None
    windows: dict[str, WindowMovers]
