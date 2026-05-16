"""
MCP Server for Nifty Index Analysis.

Provides 17 AI-callable tools for Indian stock market index data:
- Data retrieval (historical prices, live quotes, index listings)
- Analysis (CAGR, drop stats, comparisons, top movers)
- Backtesting (fixed-day SIP, dip-based SIP, optimal SIP day)
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Strategy signals (golden/death cross, drawdowns, momentum, volatility, mean reversion)
"""

__version__ = "1.0.0"

from mcp_nifty.server import mcp  
