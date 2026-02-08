import os
import sys
import django
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. DEFINE PATHS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
# In Docker, project_root is /app. We need /app/src/portal for Django.
django_root = os.path.join(project_root, 'src', 'portal')

if project_root not in sys.path: sys.path.append(project_root)
if django_root not in sys.path: sys.path.append(django_root)

# --- 2. SETUP DJANGO ---
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings') 
django.setup()

# --- 3. IMPORT VIEWS ---
try:
    from dashboard.views import sync_stock_on_demand
    print("✅ Successfully imported dashboard logic!")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def is_market_open():
    """Checks if the NSE market is currently open."""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    if now.weekday() >= 5: return False
    market_start, market_end = time(9, 0), time(16, 0)
    if not (market_start <= now.time() <= market_end): return False
    try:
        schedule = mcal.get_calendar('NSE').schedule(start_date=now.date(), end_date=now.date())
        return not schedule.empty
    except:
        return True # Fallback to open if calendar service fails

def get_comprehensive_tickers():
    """Fetches top 250+ tickers to ensure sector diversity."""
    print("📋 Fetching Comprehensive NSE Stock List...")
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        
        # Take Nifty 250 for broad coverage
        tickers = df['Symbol'].tolist()[:250] 
        
        # Ensure mission-critical sectors are present
        important_extras = ["JETAIRWAYS", "SPICEJET", "HAL", "BEL", "MAZDOCK", "RVNL", "IRFC", "ZOMATO"]
        for extra in important_extras:
            if extra not in tickers:
                tickers.append(extra)

        print(f"✅ Prepared {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"⚠️ Ticker Fetch Failed: {e}. Using Fallback.")
        return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "LT", "HINDUNILVR"]

def sync_wrapper(symbol, index, total):
    """Wrapper to handle individual stock sync inside the thread pool."""
    try:
        print(f"🔄 [{index}/{total}] Processing {symbol}...")
        sync_stock_on_demand(symbol)
        return f"✅ {symbol} synced."
    except Exception as e:
        return f"❌ Error {symbol}: {str(e)}"

def run_etl():
    print("🚀 Starting Industry-Wide Market Data ETL (Multithreaded)...")
    
    # 1. Market Check
    if len(sys.argv) > 1 and sys.argv[1] == 'force':
        print("💪 Force Mode Activated.")
    elif not is_market_open():
        print("⏰ Market is closed. Execution skipped.")
        return

    # 2. Get Tickers
    tickers = get_comprehensive_tickers()
    total = len(tickers)

    # 3. Parallel Execution (Threading)
    # Using 10 workers balances speed and API rate limits
    print(f"⚡ Starting Parallel Sync with 10 workers...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(sync_wrapper, symbol, i+1, total) for i, symbol in enumerate(tickers)]
        
        for future in as_completed(futures):
            # Print results as they finish
            print(future.result())

    print("\n" + "="*30)
    print("✅ ETL Complete. BigQuery _v3 updated.")
    print("="*30)

if __name__ == "__main__":
    run_etl()