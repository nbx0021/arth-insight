import yfinance as yf
import pandas as pd

def get_dynamic_comparison(stock_symbol, benchmark_symbol="^NSEI"):
    """
    Professional grade fetcher for Arth-Insight.
    Uses Adjusted Close for Total Return analysis.
    """
    # 1. Clean the ticker input
    if not stock_symbol.endswith(".NS") and not stock_symbol.startswith("^"):
        stock_symbol += ".NS"
        
    tickers = [stock_symbol, benchmark_symbol]
    print(f"🔍 [FINANCE] Comparing: {stock_symbol} vs {benchmark_symbol}")
    
    # 2. auto_adjust=True is critical for dividends/splits
    data = yf.download(
        tickers, 
        period="6mo", 
        interval="1d", 
        auto_adjust=True, 
        group_by='ticker'
    )
    
    result = {}
    for t in tickers:
        # DEFENSIVE CHECK: Ensure the ticker exists and has data rows
        if t in data and not data[t].dropna().empty:
            df = data[t].dropna().copy()
            
            # 3. Calculate Cumulative Return (Growth of 1 Rupee)
            # This only runs if df is NOT empty, preventing the IndexError
            df['return'] = (df['Close'] / df['Close'].iloc[0]) - 1
            
            # 4. Solve 'KeyError: Date'
            df = df.reset_index() 
            
            # 5. Format for JSON
            result[t] = df[['Date', 'Close', 'return']].to_dict('records')
        else:
            print(f"⚠️ Warning: No data found for ticker {t}")
            result[t] = []
            
    return result