import pandas as pd
import requests
from io import StringIO

def get_nse_master_list():
    """
    Downloads the full list of NSE stocks to create a dynamic mapping.
    Returns: A dictionary like {'CELLO WORLD': 'CELLO', 'RELIANCE INDUSTRIES': 'RELIANCE'}
    """
    try:
        # This is a reliable mirror of the official NSE Equity list (updated daily)
        url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        
        # We need headers to look like a real browser, or NSE blocks us
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Create a clean dictionary: { "COMPANY NAME": "SYMBOL" }
            mapping = {}
            for index, row in df.iterrows():
                # Clean the name: "Cello World Limited" -> "CELLO WORLD"
                clean_name = row['NAME OF COMPANY'].upper().replace("LIMITED", "").replace("LTD", "").replace(".", "").strip()
                symbol = row['SYMBOL']
                
                mapping[clean_name] = symbol
                mapping[symbol] = symbol  # Also map the symbol itself (TCS -> TCS)
            
            return mapping
    except Exception as e:
        print(f"Failed to fetch NSE Master List: {e}")
        return {}

    return {}


