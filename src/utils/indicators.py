import pandas as pd

def calculate_rsi(series, period=14):
    """
    Standard RSI calculation (Wilder's Smoothing).
    Proof of financial engineering logic.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Relative Strength
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi