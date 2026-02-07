import os
import sys
import io
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from google.cloud import bigquery
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# --- BOOTSTRAP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.indicators import calculate_rsi 

load_dotenv()

def get_nifty_500_tickers():
    """Fetches the official Nifty 500 list from NSE."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(response.text))
        return [f"{symbol}.NS" for symbol in df['Symbol']]
    except Exception as e:
        print(f"⚠️ NSE Fetch failed, using fallback. Error: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

def process_single_stock(symbol):
    """Worker function with extra safety checks."""
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Fetch Price History (Required for RSI)
        hist = ticker.history(period="1y")
        if hist is None or hist.empty:
            print(f"⚠️ No price history for {symbol}")
            return None

        # 2. Fetch Info (Metadata)
        info = ticker.info if ticker.info else {}

        # 3. Fetch Financials (Defensive Logic)
        # Using .income_stmt instead of .financials for better reliability
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        
        profit_margin = 0
        try:
            # Check if DataFrames are not None and not empty before accessing
            if income_stmt is not None and not income_stmt.empty:
                if 'Net Income' in income_stmt.index and 'Total Revenue' in income_stmt.index:
                    net_inc = income_stmt.loc['Net Income'].iloc[0]
                    rev = income_stmt.loc['Total Revenue'].iloc[0]
                    if rev and rev != 0:
                        profit_margin = round((net_inc / rev) * 100, 2)
        except Exception:
            profit_margin = 0 # Fallback if specific rows are missing

        # 4. Calculate RSI
        hist['RSI'] = calculate_rsi(hist['Close'])
        current_rsi = round(float(hist['RSI'].iloc[-1]), 2) if not hist['RSI'].empty else 0

        return {
            "ticker": symbol.replace(".NS", ""),
            "pe_ratio": info.get("trailingPE"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "profit_margin": profit_margin,
            "current_rsi": current_rsi,
            "market_cap": info.get("marketCap"),
            "last_updated": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"❌ Skipping {symbol} due to error: {e}")
        return None

def sync_warehouse(max_stocks=50):
    client = bigquery.Client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    table_id = f"{client.project}.{dataset_id}.stock_intelligence"

    print(f"🌍 Fetching Universe...")
    tickers = get_nifty_500_tickers()[:max_stocks]
    
    print(f"🚀 Starting Sync for {len(tickers)} stocks...")
    
    all_data = []
    # Using threads but with error isolation
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_single_stock, tickers))
    
    all_data = [r for r in results if r is not None]

    if not all_data:
        print("❌ No data collected. Check internet or ticker list.")
        return

    df = pd.DataFrame(all_data)
    # Ensure columns match BigQuery expectations
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    print(f"📤 Uploading {len(df)} rows to BigQuery...")
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"✅ Success! Warehouse updated.")
    
if __name__ == "__main__":
    sync_warehouse(max_stocks=50) # Change to 500 when you're ready for the full run!