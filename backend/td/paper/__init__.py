"""Paper trading: an automatic intraday (long/short, up to 5x) and swing (long-only) trader with a trade book.

Nothing here talks to a broker: orders are simulated against live prices (or history, in backtests) with
Indian equity charges and slippage, and every trade records why it was taken and why it was closed.
"""
