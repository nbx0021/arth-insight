import os
import sys
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, time
import pytz

# --- SETUP DJANGO ENVIRONMENT ---
# This allows us to use your existing 'views.py' logic without running the server
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arth_insight.settings') 
import django
django.setup()

# Now we can import your core logic!
from dashboard.views import sync_stock_on_demand

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
    """Fetches Dynamic Nifty 50 List from NSE Website."""
    print("📋 Fetching Nifty 50 list from NSE...")
    try:
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
        df = pd.read_csv(url)
        tickers = df['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"⚠️ Failed to fetch live list ({e}). Using Fallback.")
        # Fallback to Top 20 Heavyweights if NSE site is slow/blocking
        return [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", 
            "SBIN", "BHARTIARTL", "LTIM", "AXISBANK", "KOTAKBANK",
            "LT", "HINDUNILVR", "TATAMOTORS", "M&M", "MARUTI",
            "TITAN", "SUNPHARMA", "BAJFINANCE", "ASIANPAINT"
        ]

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