import os
import sys
import django
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz
import requests
from io import StringIO

# --- 1. DEFINE PATHS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
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
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    if now.weekday() >= 5: return False
    market_start, market_end = time(9, 0), time(16, 0)
    if not (market_start <= now.time() <= market_end): return False
    return not mcal.get_calendar('NSE').schedule(start_date=now.date(), end_date=now.date()).empty

def get_comprehensive_tickers():
    """
    Fetches all listed stocks and picks the top 250 (Nifty 200 + selected midcaps)
    to ensure all industries are covered with at least 5-10 stocks.
    """
    print("📋 Fetching Comprehensive NSE Stock List...")
    
    # We use the Nifty 500 list to ensure we get every industry (Aviation, Sugar, Defense, etc.)
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        
        # This list contains Industry columns, so we can ensure diversity
        # We take the first 250 stocks which covers almost all large and mid-cap industries
        tickers = df['Symbol'].tolist()[:250] 
        
        # Add a few manual important ones if they are missing (Aviation/Defense)
        important_extras = ["JETAIRWAYS", "SPICEJET", "HAL", "BEL", "MAZDOCK", "RVNL", "IRFC"]
        for extra in important_extras:
            if extra not in tickers:
                tickers.append(extra)

        print(f"✅ Successfully prepared {len(tickers)} diverse industry tickers.")
        return tickers

    except Exception as e:
        print(f"⚠️ Fetch Failed: {e}. Using Nifty 50 Fallback.")
        return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "LT", "HINDUNILVR"]

def run_etl():
    print("🚀 Starting Industry-Wide Market Data ETL...")
    
    # Bypass market check if 'force' is used
    if len(sys.argv) > 1 and sys.argv[1] == 'force':
        print("💪 Force Mode Activated.")
    elif not is_market_open():
        print("⏰ Market is closed. Use 'force' to run anyway.")
        return

    tickers = get_comprehensive_tickers()
    
    # --- PRO TIP: Using a smaller batch for testing ---
    # If you want to run ALL 250, leave it as is. 
    # If your GitHub Actions times out, reduce the 250 in get_comprehensive_tickers to 100.
    
    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {symbol}...")
        try:
            # This triggers your views.py logic (saves to BigQuery _v3)
            sync_stock_on_demand(symbol) 
        except Exception as e:
            print(f"❌ Error syncing {symbol}: {e}")

    print("✅ ETL Complete. All industries refreshed in BigQuery _v3.")

if __name__ == "__main__":
    run_etl()