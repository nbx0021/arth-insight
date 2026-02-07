import os
import pandas as pd
import yfinance as yf
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
import pytz
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "SBIN",
    "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA", 
    "TATAMOTORS", "AXISBANK", "ONGC", "TITAN", "KOTAKBANK", "ADANIPORTS", 
    "COALINDIA", "ASIANPAINT", "TATASTEEL", "ULTRACEMCO", "JSWSTEEL", "M&M"
]

def perform_fresh_start():
    BASE_DIR = Path(__file__).resolve().parent
    key_path = BASE_DIR / "service-account.json"
    
    credentials = service_account.Credentials.from_service_account_file(str(key_path))
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset_id = os.getenv("GCP_DATASET_ID", "stock_raw_data")
    table_id = f"{client.project}.{dataset_id}.stock_intelligence"

    all_stock_data = []

    print(f"🚀 Starting Bulk Sync for {len(NIFTY_50)} stocks... for testing")
    
    for symbol in NIFTY_50:
        try:
            print(f"📊 Fetching {symbol}...")
            ticker_obj = yf.Ticker(f"{symbol}.NS")
            info = ticker_obj.info
            
            # Simple RSI Calculation
            hist = ticker_obj.history(period="1mo")
            rsi = 50 # Default
            if len(hist) > 14:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))

            all_stock_data.append({
                'ticker': symbol,
                'sector': info.get('sector', 'Unknown'),
                'pe_ratio': info.get('trailingPE'),
                'current_rsi': round(float(rsi), 2),
                'profit_margin': round(info.get('profitMargins', 0) * 100, 2),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': round(info.get('returnOnEquity', 0) * 100, 2),
                'last_updated': datetime.now(pytz.UTC)
            })
            
        except Exception as e:
            print(f"⚠️ Error syncing {symbol}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_stock_data)

    # STEP 2: OVERWRITE TABLE (This works in Free Tier!)
    print(f"📡 Uploading to BigQuery and Overwriting existing data...")
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", # This deletes old and adds new automatically
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result() # Wait for job to complete

    print("🏁 Bulk Sync Complete. Table has been completely refreshed!")

if __name__ == "__main__":
    perform_fresh_start()