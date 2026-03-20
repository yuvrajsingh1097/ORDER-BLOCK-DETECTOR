Order Blocks are defined by what comes after the candle, not just the candle itself — this makes them a forward-looking detection problem
ATR normalization is essential: a 1-point move on NQ futures means nothing; a 1-pip move on EURUSD is significant. Using ATR multiples makes the detector work across any instrument
Tested OBs have a higher probability of holding as support/resistance than fresh ones — but mitigated OBs flip to the opposite bias (this is the ICT Breaker Block concept)
Tracking 3 distinct states (fresh / tested / mitigated) teaches you how to implement a simple state machine on time-series data


Tech stack

yfinance — OHLCV data
pandas + numpy — detection and status logic
matplotlib — candlestick chart with overlaid OB zones
mplfinance — helper for financial chart styling
pytest — unit tests for all core functions
