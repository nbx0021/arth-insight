import os
import sys
import django
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz
import requests
from io import StringIO

# --- 1. DEFINE PATHS (The Fix) ---
# Current Script: .../arth-insight/scripts/market_feed.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Project Root: .../arth-insight (This allows "from src.utils import...")
project_root = os.path.dirname(current_script_dir)

# Django Root: .../arth-insight/src/portal (This is where 'config' and 'dashboard' live)
# ERROR WAS HERE: You had 'src', it needs to be 'src/portal'
django_root = os.path.join(project_root, 'src', 'portal')

# --- 2. ADD PATHS TO SYSTEM ---
# We add BOTH so Python can find 'dashboard' AND 'src.utils'
if project_root not in sys.path:
    sys.path.append(project_root)

if django_root not in sys.path:
    sys.path.append(django_root)

# --- 3. SETUP DJANGO ---
# Now Python can find 'config' inside 'src/portal'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings') 
django.setup()

# --- 4. IMPORT VIEWS ---
try:
    # Now we import directly from 'dashboard' because we are "standing" inside 'portal'
    from dashboard.views import sync_stock_on_demand
    print("✅ Successfully imported dashboard logic!")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def is_market_open():
    """
    Checks if NSE is currently open (holidays + weekend + time check).
    Returns True if Market is Open, False otherwise.
    """
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    
    # 1. Check Weekend (Saturday=5, Sunday=6)
    if now.weekday() >= 5: 
        print(f"📅 Today is Weekend ({now.strftime('%A')}). Market Closed.")
        return False

    # 2. Check Time (9:00 AM - 4:00 PM buffer)
    market_start = time(9, 0)
    market_end = time(16, 0)
    current_time = now.time()
    
    if not (market_start <= current_time <= market_end):
        print(f"⏰ Outside Market Hours ({current_time.strftime('%H:%M')}). Skipping.")
        return False

    # 3. Check NSE Holidays
    nse = mcal.get_calendar('NSE')
    schedule = nse.schedule(start_date=now.date(), end_date=now.date())
    if schedule.empty:
        print("🏖️ Today is a Market Holiday.")
        return False
        
    return True



def get_nifty50_tickers():
    """Fetches Dynamic Nifty 50 List from NSE Website with Anti-Blocking."""
    print("📋 Fetching Nifty 50 list from NSE...")
    
    fallback_tickers = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", 
        "SBIN", "BHARTIARTL", "LTIM", "AXISBANK", "KOTAKBANK",
        "LT", "HINDUNILVR", "TATAMOTORS", "M&M", "MARUTI",
        "TITAN", "SUNPHARMA", "BAJFINANCE", "ASIANPAINT"
    ]

    try:
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
        
        # 1. TRICK NSE: Pretend to be a Chrome Browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 2. FETCH WITH TIMEOUT: Don't wait more than 10 seconds
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Check for 404/500 errors
        
        # 3. CONVERT TO DATAFRAME
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        tickers = df['Symbol'].tolist()
        print(f"✅ Successfully fetched {len(tickers)} tickers from NSE.")
        return tickers

    except Exception as e:
        print(f"⚠️ NSE Fetch Failed or Timed Out ({e}).")
        print("🔄 Switching to Fallback List (Top 20 Heavyweights).")
        return fallback_tickers

def run_etl():
    print("🚀 Starting Market Data ETL...")
    
    # 1. Check Market Status
    # (Allow 'force' argument to bypass check for testing)
    if len(sys.argv) > 1 and sys.argv[1] == 'force':
        print("💪 Force Mode Activated.")
    elif not is_market_open():
        return

    # 2. Get Stocks
    tickers = get_nifty50_tickers()
    print(f"✅ Found {len(tickers)} stocks to sync.")

    # 3. Sync Each Stock
    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Syncing {symbol}...")
        try:
            # We pass the raw symbol. 'sync_stock_on_demand' handles the ".NS" logic.
            sync_stock_on_demand(symbol) 
        except Exception as e:
            print(f"❌ Error syncing {symbol}: {e}")

    print("✅ ETL Complete. Database updated.")

if __name__ == "__main__":
    run_etl()