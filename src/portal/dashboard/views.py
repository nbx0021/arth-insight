import os, json, pytz
from xmlrpc import client
import pandas as pd
import yfinance as yf
import numpy as np
import feedparser
from datetime import datetime, timedelta
from django.shortcuts import render
from pathlib import Path
from google.cloud import bigquery
from google.oauth2 import service_account
from difflib import get_close_matches
from dotenv import load_dotenv
import pytz
from textblob import TextBlob
from django.core.cache import cache
from django.utils.text import slugify

# Custom Utilities
from src.utils.finance_provider import get_dynamic_comparison
from src.utils.indicators import calculate_rsi 

from .utils import get_nse_master_list
from django.core.cache import cache

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================

# --- 1. TOP 50 INDIAN STOCKS MAPPING (Name -> Ticker) ---
# eg: This allows users to search by "Reliance" instead of just "RELIANCE"
COMPANY_TO_TICKER_MAP = {
    "RELIANCE": "RELIANCE", "RELIANCE INDUSTRIES": "RELIANCE",
    "TCS": "TCS", "TATA CONSULTANCY SERVICES": "TCS",
    "HDFC": "HDFCBANK", "HDFC BANK": "HDFCBANK",
    "ICICI": "ICICIBANK", "ICICI BANK": "ICICIBANK",
    "INFY": "INFY", "INFOSYS": "INFY",
    "ITC": "ITC",
    "SBI": "SBIN", "STATE BANK OF INDIA": "SBIN",
    "BHARTI": "BHARTIARTL", "AIRTEL": "BHARTIARTL",
    "HUL": "HINDUNILVR", "HINDUSTAN UNILEVER": "HINDUNILVR",
    "L&T": "LT", "LARSEN": "LT", "LARSEN & TOUBRO": "LT",
    "TATA MOTORS": "TATAMOTORS", "TATAMOTORS": "TATAMOTORS",
    "M&M": "M&M", "MAHINDRA": "M&M",
    "MARUTI": "MARUTI", "MARUTI SUZUKI": "MARUTI",
    "ADANI ENT": "ADANIENT", "ADANI ENTERPRISES": "ADANIENT",
    "ADANI PORTS": "ADANIPORTS",
    "SUN PHARMA": "SUNPHARMA",
    "ASIAN PAINTS": "ASIANPAINT",
    "BAJAJ FINANCE": "BAJFINANCE",
    "BAJAJ FINSERV": "BAJAJFINSV",
    "TITAN": "TITAN",
    "WIPRO": "WIPRO",
    "ULTRATECH": "ULTRACEMCO",
    "NTPC": "NTPC",
    "ONGC": "ONGC",
    "POWERGRID": "POWERGRID",
    "JSW STEEL": "JSWSTEEL",
    "TATA STEEL": "TATASTEEL",
    "HCL": "HCLTECH", "HCL TECH": "HCLTECH",
    "COAL INDIA": "COALINDIA",
    "ZOMATO": "ZOMATO",
    "PAYTM": "PAYTM",
    "NYKAA": "NYKAA",
    "DMART": "DMART", "AVENUE SUPERMARTS": "DMART",
    "VBL": "VBL", "VARUN BEVERAGES": "VBL",
    "JIO": "JIOFIN", "JIO FINANCIAL": "JIOFIN"
}

# 1. YOUR MANUAL "CHEAT SHEET" (Keep this!)
# Rename it to CUSTOM_ALIASES to be clear it's for special names
CUSTOM_ALIASES = {
    "PAYTM": "PAYTM",
    "HDFC": "HDFCBANK",
    "BAJAJ FINANCE": "BAJFINANCE",
    "L&T": "LT",
    "JIO": "JIOFIN",
    "RELIANCE": "RELIANCE",
    # --- ADD THIS NEW LINE ---
    "CROMPTON": "CROMPTON",
    "CROMPTON GREAVES": "CROMPTON", 
    "CROMPTON GREAVES CONSUMER ELECTRICALS": "CROMPTON",
    # --- FIX FOR SPICEJET ---
    # Yahoo often breaks 'SPICEJET.NS', so we use the BSE Code (500285.BO)
    "SPICEJET": "500285.BO",
    "SPICE JET": "500285.BO",
}

# --- LOAD MAPPING ON STARTUP ---
# We try to get it from cache first, if not, we fetch it (takes 1 second)
TICKER_MAPPING = cache.get('nse_master_mapping')

if not TICKER_MAPPING:
    print("🌍 Downloading latest NSE Stock List...")
    TICKER_MAPPING = get_nse_master_list()
    # Save to cache for 24 hours so we don't download it every time
    cache.set('nse_master_mapping', TICKER_MAPPING, timeout=60*60*24)

def resolve_symbol_from_name(query):
    # 1. CLEAN THE INPUT (Aggressive Cleaning)
    # We remove "LTD", "LIMITED", "INDIA", dots, and extra spaces.
    # This turns "Crompton... Ltd." -> "CROMPTON GREAVES CONSUMER ELECTRICALS"
    clean_query = query.upper()
    for word in [".", " LTD", " LIMITED", " INDIA", " PVT", " PRIVATE", " INC"]:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()
    
    # 2. CHECK MANUAL ALIASES (Your Cheat Sheet)
    if clean_query in CUSTOM_ALIASES:
        return CUSTOM_ALIASES[clean_query]
    
    # 3. CHECK DYNAMIC MASTER LIST (From utils.py)
    from .utils import get_nse_master_list
    from django.core.cache import cache

    # Fetch mapping
    full_mapping = cache.get('nse_master_mapping')
    if not full_mapping:
        full_mapping = get_nse_master_list()
        cache.set('nse_master_mapping', full_mapping, timeout=60*60*24)
    
    # 4. SEARCH LOGIC (The Fix)
    if full_mapping:
        # A. Direct Match
        if clean_query in full_mapping:
            return full_mapping[clean_query]
            
        # B. Partial Match (Loop through the list)
        # We check if your clean query is inside the company name
        for company_name, ticker in full_mapping.items():
            # Clean the company name from the list too!
            clean_company_name = company_name.replace("LIMITED", "").replace("LTD", "").strip()
            
            if clean_query in clean_company_name: 
                # FOUND IT! (e.g. "CROMPTON..." in "CROMPTON... LIMITED")
                return ticker

    # 5. Fallback
    return clean_query

SECTOR_BENCHMARKS = {
    "TECHNOLOGY": "^CNXIT", "IT SERVICES": "^CNXIT", "NEW AGE TECH": "^CNXIT",
    "FINANCIAL SERVICES": "^NSEBANK", "BANKING": "^NSEBANK",
    "ENERGY": "^CNXENERGY", 
    "CONSUMER CYCLICAL": "^CNXAUTO", "AUTO": "^CNXAUTO",
    "CONSUMER DEFENSIVE": "^CNXFMCG", "FMCG": "^CNXFMCG",
    "BASIC MATERIALS": "^CNXMETAL", "METALS": "^CNXMETAL",
    "HEALTHCARE": "^CNXPHARMA"
}

FALLBACK_SECTOR_PE = {
    "TECHNOLOGY": 28.0, "IT SERVICES": 26.0, "NEW AGE TECH": 50.0,
    "BANKING": 18.0, "FINANCIAL SERVICES": 20.0,
    "AUTO": 22.0, "ENERGY": 12.0, "FMCG": 45.0, "METALS": 15.0,
    "Unknown": 20.0
}

TICKER_SECTOR_MAP = {
    "ZOMATO": "NEW AGE TECH", "PAYTM": "NEW AGE TECH", "NYKAA": "NEW AGE TECH",
    "TATAMOTORS": "AUTO", "M&M": "AUTO", "MARUTI": "AUTO",
    "HDFCBANK": "BANKING", "SBIN": "BANKING", "ICICIBANK": "BANKING",
    "RELIANCE": "ENERGY"
}

RATING_THRESHOLDS = {
    'PROMOTER_GOOD': 50.0, 'PROMOTER_AVG': 30.0,
    'ROE_GOOD': 15.0, 'ROE_AVG': 8.0,
    'DEBT_SAFE': 0.1, 'DEBT_RISKY': 1.0,
    'VALUATION_FAIR_BUFFER': 1.3
}

# ==========================================
# 2. CORE HELPERS & DB
# ==========================================

# this refine sector from the yahoo description to get a more accurate sector for the indian context. 
# This is used in the views to classify the stock into the right sector for comparison and policy context.
def refine_sector_name(yahoo_sector, yahoo_industry, ticker, description=""):
    sec = str(yahoo_sector).upper()
    ind = str(yahoo_industry).upper()
    
    # --- 1. CLEAN THE TICKER (The Critical Fix) ---
    # Remove .NS and .BO so "ITC.NS" becomes "ITC"
    sym = str(ticker).upper().replace(".NS", "").replace(".BO", "").strip()
    
    # DEBUG: Print what the system sees
    print(f"🕵️ SECTOR CHECK: Input='{ticker}' -> Cleaned='{sym}'")

    desc = str(description).upper() if description else ""

    # =========================================================
    # 2. GOD MODE: HARDCODED OVERRIDES (Highest Priority)
    # =========================================================
    
    # FMCG (Force ITC here - MUST BE FIRST)
    if sym in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "MARICO", "VBL", "VARUN BEVERAGES"]: 
        print(f"✅ God Mode Triggered: {sym} -> FMCG")
        return "FMCG"

    # Oil & Gas (Reliance)
    if sym in ["RELIANCE", "ONGC", "OIL", "IOC", "BPCL", "HPCL", "GAIL"]: 
        return "OIL & GAS"
    
    # Aviation (Airlines)
    if sym in ["INDIGO", "SPICEJET", "JETAIRWAYS", "GLOBALVECT"]: 
        return "AVIATION"

    # Infrastructure & Logistics
    if sym in ["ADANIPORTS", "GPPL", "JSWINFRA", "IRB", "GMRINFRA", "L&T", "LT"]: 
        return "INFRASTRUCTURE"
    
    if sym in ["CONCOR", "VRL", "TCI", "BLUE DART", "BLUEDART"]: 
        return "LOGISTICS"

    # Power & Renewables
    if sym in ["SUZLON", "INOXWIND", "KPIGREEN", "ADANIGREEN", "TATAPOWER", "NTPC", "POWERGRID", "SJVN", "NHPC"]: 
        return "POWER & RENEWABLES"
    
    # Consumer Durables
    if sym in ["BLUESTARCO", "VOLTAS", "WHIRLPOOL", "CROMPTON", "HAVELLS", "POLYCAB", "DIXON", "AMBER", "KAJARIA"]: 
        return "CONSUMER DURABLES"

    # Capital Goods / Engineering
    if sym in ["HONAUT", "ABB", "SIEMENS", "THERMAX", "CUMMINSIND", "BHEL"]: 
        return "CAPITAL GOODS"
    
    # Consumer Tech
    if sym in ["ZOMATO", "PAYTM", "NYKAA", "POLICYBZR", "DELHIVERY", "NAUKRI"]: 
        return "CONSUMER TECH"

    # IT Services 
    if sym in ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS"]:
        return "IT SERVICES"

    # =========================================================
    # 3. DYNAMIC SCANNING (Keyword Search)
    # =========================================================
    
    # AVIATION (Check this FIRST to catch any new airlines)
    if "AIRLINE" in desc or "AVIATION" in desc or "PASSENGER AIRCRAFT" in desc:
        return "AVIATION"

    # INFRASTRUCTURE
    if "PORTS" in desc or "HIGHWAY" in desc or "TOLL" in desc or "CONSTRUCTION PROJECT" in desc or "EPC" in desc:
        return "INFRASTRUCTURE"

    # LOGISTICS (Check this AFTER Aviation so "Air Cargo" doesn't trigger it)
    if "LOGISTICS" in desc or "CARGO" in desc or "FREIGHT" in desc or "WAREHOUS" in desc or "TRANSPORT" in desc:
        return "LOGISTICS"

    # POWER
    if "WIND ENERGY" in desc or "SOLAR" in desc or "HYDRO" in desc or "THERMAL POWER" in desc:
        return "POWER & RENEWABLES"

    # CONSUMER DURABLES
    if "AIR CONDITION" in desc or "REFRIGERATOR" in desc or "CABLES" in desc or "TILES" in desc:
        return "CONSUMER DURABLES"

    # DEFENSE
    if "DEFENSE" in desc or "MISSILE" in desc or "NAVAL" in desc or "AEROSPACE" in desc:
        return "DEFENSE"

    
    # =========================================================
    # 4. FALLBACK TO STANDARD MAPPINGS
    # =========================================================
    if "BANK" in ind: return "BANKING"
    if "TECHNOLOGY" in sec: return "IT SERVICES"
    if "PHARMA" in ind: return "PHARMA"
    if "AUTO" in ind: return "AUTOMOBILE"
    if "REAL ESTATE" in sec: return "REALTY"
    if "FMCG" in sec or "FOOD" in ind: return "FMCG"

    return sec

def get_bq_client():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    key_path = BASE_DIR / "service-account.json"
    credentials = service_account.Credentials.from_service_account_file(str(key_path))
    return bigquery.Client(credentials=credentials, project=credentials.project_id)

def to_crores(value):
    try: return round(float(value) / 10000000, 2) if value else 0
    except: return 0

def format_shares(value):
    try:
        val = float(value)
        if val > 1_000_000_000: return f"{round(val / 1_000_000_000, 2)} B"
        if val > 10_000_000: return f"{round(val / 10_000_000, 2)} Cr"
        return str(value)
    except: return "N/A"

def safe_float(val):
    try:
        if hasattr(val, 'iloc'): return float(val.iloc[0])
        if hasattr(val, 'item'): return float(val.item())
        if val is None or pd.isna(val): return 0.0
        return float(val)
    except: return 0.0

def calculate_cagr(start_price, end_price, years):
    if start_price == 0 or years == 0: return 0.0
    return round(((end_price / start_price) ** (1 / years) - 1) * 100, 2)

def get_industry_pe(sector, ticker, fallback_val):
    try:
        client = get_bq_client()
        dataset_id = os.getenv("GCP_DATASET_ID")
        # NEW (Adds _v2 to create a fresh table)
        table_id = f"{client.project}.{dataset_id}.stock_intelligence_v2"
        query = f"SELECT AVG(pe_ratio) as avg_pe FROM `{table_id}` WHERE sector = '{sector}' AND ticker != '{ticker}' AND pe_ratio > 0"
        df = client.query(query).to_dataframe()
        if not df.empty and pd.notnull(df['avg_pe'].iloc[0]):
            val = float(df['avg_pe'].iloc[0])
            if val > 0: return round(val, 2)
    except: pass
    return fallback_val



def get_accounting_row(df, keys, label, indent=0, bg_color=None, bold=False, negative=False):
    style_str = f"padding-left: {indent * 20 + 10}px;"
    if bg_color: style_str += f" background-color: {bg_color};"
    if bold: style_str += " font-weight: 700;"
    
    vals = [0, 0]
    for k in keys:
        if k in df.index:
            raw_vals = df.loc[k].values[:2]
            fmt_vals = []
            for v in raw_vals:
                val_cr = round(v / 10000000, 2)
                if negative and val_cr > 0: val_cr = -val_cr
                fmt_vals.append(val_cr)
            while len(fmt_vals) < 2: fmt_vals.append(0)
            vals = fmt_vals
            break
    return {'label': label, 'values': vals, 'style': style_str}

# --- NEW: Technical Analysis Engine ---
def calculate_technicals(hist):
    """
    Calculates key technical indicators from historical data.
    """
    if hist.empty: return None
    
    # 1. Simple Moving Averages (SMA)
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    
    # 2. RSI (Relative Strength Index) - 14 Days
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (Moving Average Convergence Divergence)
    # EMA 12 - EMA 26
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = ema12 - ema26
    hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Get Latest Values
    latest = hist.iloc[-1]
    
    # 5. Interpret Signals
    signals = {}
    
    # RSI Signal
    rsi = round(latest['RSI'], 1)
    if rsi > 70: signals['rsi'] = {'val': rsi, 'status': 'Overbought', 'color': 'text-danger'}
    elif rsi < 30: signals['rsi'] = {'val': rsi, 'status': 'Oversold', 'color': 'text-success'}
    else: signals['rsi'] = {'val': rsi, 'status': 'Neutral', 'color': 'text-warning'}
    
    # SMA Trend
    price = latest['Close']
    sma50 = latest['SMA_50']
    sma200 = latest['SMA_200']
    
    if price > sma50 and price > sma200: 
        signals['trend'] = {'status': 'Strong Bullish', 'color': 'text-success', 'desc': 'Price > 50 & 200 DMA'}
    elif price < sma50 and price < sma200:
        signals['trend'] = {'status': 'Strong Bearish', 'color': 'text-danger', 'desc': 'Price < 50 & 200 DMA'}
    elif price > sma200:
        signals['trend'] = {'status': 'Bullish Bias', 'color': 'text-primary', 'desc': 'Price > 200 DMA'}
    else:
        signals['trend'] = {'status': 'Bearish Bias', 'color': 'text-warning', 'desc': 'Price < 200 DMA'}

    # MACD Signal
    macd = latest['MACD']
    sig = latest['Signal_Line']
    if macd > sig: signals['macd'] = {'val': round(macd, 2), 'status': 'Bullish Crossover', 'color': 'text-success'}
    else: signals['macd'] = {'val': round(macd, 2), 'status': 'Bearish Crossover', 'color': 'text-danger'}

    return signals


# --- ROBUST NEWS ENGINE ---
def get_robust_news(ticker_obj, symbol):
    news_data = []
    # 1. Try Yahoo Finance First
    try:
        raw_news = ticker_obj.news
        if raw_news:
            for n in raw_news[:5]:
                title = n.get('title') or n.get('headline')
                if not title: continue
                ts = n.get('providerPublishTime') or n.get('pubDate') or 0
                date_str = datetime.fromtimestamp(ts).strftime('%d %b %Y') if ts else "Recent"
                news_data.append({
                    'title': title,
                    'link': n.get('link') or n.get('url'),
                    'publisher': n.get('publisher') or n.get('provider') or "Yahoo Finance",
                    'date': date_str,
                    'source': 'Yahoo'
                })
    except: pass

    # 2. Fallback to Google News RSS
    if not news_data:
        try:
            query = f"{symbol} stock news India"
            encoded_query = query.replace(" ", "%20")
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                try:
                    dt = datetime.strptime(entry.published[:16], '%a, %d %b %Y')
                    date_str = dt.strftime('%d %b %Y')
                except: date_str = "Recent"
                news_data.append({
                    'title': entry.title,
                    'link': entry.link,
                    'publisher': entry.source.title if hasattr(entry, 'source') else "Google News",
                    'date': date_str,
                    'source': 'Google'
                })
        except Exception as e: print(f"Google News Error: {e}")
    return news_data

# --- POLICY MAPPER (FIXED PATH LOGIC) ---
def get_policy_context(sector_name, industry_name):
    try:
        # Navigate from: src/portal/dashboard/views.py -> src/data/policy_context.json
        # .parent = dashboard
        # .parent.parent = portal
        # .parent.parent.parent = src (Root of source code)
        base_path = Path(__file__).resolve().parent.parent.parent
        file_path = base_path / 'src' / 'data' / 'policy_context.json'
        
        # Fallback if 'src' is not repeated in folder structure
        if not file_path.exists():
             file_path = base_path / 'data' / 'policy_context.json'

        with open(file_path, 'r') as f:
            policy_data = json.load(f)
            
    except Exception as e:
        print(f"Policy Load Error: {e} | Looking at: {file_path}")
        return None

    sec = str(sector_name).upper()
    ind = str(industry_name).upper()
    target_key = None

    # Enhanced Logic Mapper
    if "TEXTILE" in sec or "APPAREL" in sec or "CLOTHING" in ind or "COTTON" in ind: target_key = "TEXTILES"
    elif "AGRI" in sec or "FARM" in ind or "FERTILIZER" in ind or "PESTICIDE" in ind: target_key = "AGRICULTURE"
    elif "RAIL" in ind or "WAGON" in ind: target_key = "RAILWAYS"
    elif "DEFENCE" in ind or "AEROSPACE" in ind or "SHIP" in ind: target_key = "DEFENCE"
    elif "CHEMICAL" in sec or "CHEMICAL" in ind: target_key = "CHEMICALS"
    elif "BANK" in sec or "BANK" in ind or "FINANCE" in ind: target_key = "BANKING"
    elif "AUTO" in sec or "AUTO" in ind or "VEHICLE" in ind: target_key = "AUTO"
    elif "TECHNOLOGY" in sec or "SOFTWARE" in ind or "IT SERVICES" in ind: target_key = "IT"
    elif "ENERGY" in sec or "OIL" in ind or "POWER" in ind or "UTILITIES" in sec: target_key = "POWER"
    elif "CONSTRUCTION" in sec or "REAL ESTATE" in ind or "INFRA" in ind or "CEMENT" in ind: target_key = "INFRA"
    elif "PHARMA" in sec or "DRUG" in ind or "BIOTECH" in ind: target_key = "PHARMA"
    elif "CONSUMER" in sec or "FMCG" in ind or "FOOD" in ind: target_key = "FMCG"
    elif "REAL ESTATE" in sec or "REIT" in ind: target_key = "REALTY"
    elif "NBFC" in ind or "CAPITAL" in ind: target_key = "NBFC"

    if target_key and target_key in policy_data['sectors']:
        return {
            'sector_name': target_key,
            'repo_rate': policy_data.get('repo_rate', 'N/A'),
            'stance': policy_data.get('policy_stance', 'N/A'),
            'dates': policy_data.get('dates', {}),
            'insights': policy_data['sectors'][target_key]
        }
    return None

# ==========================================
# 3. ANALYSIS ENGINES
# ==========================================
def analyze_financial_health(ticker_obj):
    analysis = []
    try:
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        pl = ticker_obj.income_stmt
        if bs.empty or cf.empty or pl.empty: return []

        latest_cfo = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
        latest_ni = pl.loc['Net Income'].iloc[0] if 'Net Income' in pl.index else 0
        if latest_cfo > latest_ni:
            analysis.append({"title": "High Quality Earnings", "status": "Good", "desc": "Operating Cash Flow > Net Income."})
        else:
            analysis.append({"title": "Aggressive Accounting?", "status": "Caution", "desc": "Net Income > Cash Flow."})

        total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        if cash > total_debt:
            analysis.append({"title": "Balance Sheet Fortress", "status": "Good", "desc": "Cash > Total Debt."})
        else:
            analysis.append({"title": "Leveraged Balance Sheet", "status": "Neutral", "desc": f"Net Debt: {to_crores(total_debt - cash)} Cr."})
    except: pass
    return analysis


# --- SCORING ENGINE FUNCTION ---
def calculate_arth_score(financials, technicals, policy, news_list):
    """
    The Brain: Calculates a 0-100 Health Score & Recommendation.
    Weights: Financials (40%), Technicals (30%), Macro (20%), Sentiment (10%)
    """
    score = 0
    reasons = []

    # 1. FINANCIALS (Max 40 Points)
    # ROE > 15% (+15), Profit Growth > 10% (+15), Low Debt (+10)
    if financials.get('roe', 0) > 15: score += 15
    if financials.get('profit_growth', 0) > 10: score += 15
    if financials.get('debt_to_equity', 0) < 1.0: score += 10
    else: reasons.append("High Debt Levels")

    # 2. TECHNICALS (Max 30 Points)
    # RSI not overbought/sold (+10), Bullish Trend (+20)
    if technicals:
        rsi = technicals['rsi']['val']
        if 35 < rsi < 70: score += 10
        
        trend = technicals['trend']['status']
        if "Bullish" in trend: score += 20
        elif "Bearish" in trend: score -= 5  # Penalty

    # 3. MACRO / POLICY (Max 20 Points)
    # NLP on Budget Stance
    if policy and 'insights' in policy:
        budget_text = policy['insights'].get('budget', '').lower()
        if "positive" in budget_text or "growth" in budget_text or "incentive" in budget_text:
            score += 20
        elif "neutral" in budget_text:
            score += 10

    # 4. NEWS SENTIMENT (Max 10 Points)
    # Analyze last 5 headlines
    sentiment_score = 0
    if news_list:
        for n in news_list:
            blob = TextBlob(n['title'])
            sentiment_score += blob.sentiment.polarity # Returns -1 to 1
        
        # If avg sentiment is positive
        if sentiment_score > 0: score += 10
        elif sentiment_score < -0.5: score -= 5 # Penalty for bad news

    # 5. FINAL VERDICT
    if score >= 80: verdict = "STRONG BUY"
    elif score >= 60: verdict = "BUY"
    elif score >= 40: verdict = "HOLD"
    else: verdict = "SELL / AVOID"

    return {
        'score': score,
        'verdict': verdict,
        'color': 'text-success' if score >= 60 else 'text-danger' if score < 40 else 'text-warning'
    }

def calculate_narendra_rating(data):
    ratings = {}
    thresh = RATING_THRESHOLDS
    prom = data.get('promoter_holding', 0)
    if prom > thresh['PROMOTER_GOOD']: ratings['ownership'] = {"metric": f"{prom}%", "benchmark": f"> {thresh['PROMOTER_GOOD']}%", "status": "High", "color": "text-success"}
    elif prom > thresh['PROMOTER_AVG']: ratings['ownership'] = {"metric": f"{prom}%", "benchmark": f"> {thresh['PROMOTER_AVG']}%", "status": "Stable", "color": "text-primary"}
    else: ratings['ownership'] = {"metric": f"{prom}%", "benchmark": f"> {thresh['PROMOTER_AVG']}%", "status": "Low", "color": "text-danger"}

    pe = data.get('pe_ratio', 0)
    ind_pe = data.get('industry_pe', 25)
    if pe > 0 and pe < ind_pe: ratings['valuation'] = {"metric": f"{pe}x", "benchmark": f"{ind_pe}x", "status": "Cheap", "color": "text-success"}
    elif pe < (ind_pe * thresh['VALUATION_FAIR_BUFFER']): ratings['valuation'] = {"metric": f"{pe}x", "benchmark": f"{ind_pe}x", "status": "Fair", "color": "text-primary"}
    else: ratings['valuation'] = {"metric": f"{pe}x", "benchmark": f"{ind_pe}x", "status": "Expensive", "color": "text-danger"}

    roe = data.get('roe', 0)
    if roe > thresh['ROE_GOOD']: ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_GOOD']}%", "status": "Excellent", "color": "text-success"}
    elif roe > thresh['ROE_AVG']: ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_AVG']}%", "status": "Average", "color": "text-warning"}
    else: ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_AVG']}%", "status": "Poor", "color": "text-danger"}

    de = data.get('debt_to_equity', 0)
    if de < thresh['DEBT_SAFE']: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": f"< {thresh['DEBT_SAFE']}", "status": "Debt Free", "color": "text-success"}
    elif de < thresh['DEBT_RISKY']: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": f"< {thresh['DEBT_RISKY']}", "status": "Stable", "color": "text-primary"}
    else: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": f"< {thresh['DEBT_RISKY']}", "status": "High Debt", "color": "text-danger"}
    return ratings

def calculate_wealth_growth(ticker_obj, investment_amount, start_year):
    try:
        # 1. Fetch History
        hist = ticker_obj.history(period="max")
        
        # Immediate safety check
        if hist is None or hist.empty: 
            return None

        # --- SMART TIMEZONE MATCHER ---
        # We want to keep 'Asia/Kolkata' if it exists.
        
        target_date = pd.Timestamp(f"{start_year}-01-01")
        
        # Check if the downloaded data has a timezone
        if hist.index.tz is not None:
            # YES: It has a timezone (e.g., Asia/Kolkata).
            # So we apply the SAME timezone to our target date.
            target_date = target_date.tz_localize(hist.index.tz)
        
        # Now filtering is safe because both sides match (Apple vs Apple)
        hist_filtered = hist[hist.index >= target_date]
        # -----------------------------

        is_adjusted = False
        actual_start_year = start_year
        
        # Handle case where stock is younger than the requested year
        # (e.g., Asking for 2011 data for a company listed in 2015)
        if hist_filtered.empty:
            hist_filtered = hist  # Fallback to full history
            is_adjusted = True
            actual_start_year = hist.index[0].year
        
        # Double check validity after fallback
        if hist_filtered.empty:
            return None

        # 2. Wealth Calculation
        start_price = hist_filtered['Close'].iloc[0]
        current_price = hist_filtered['Close'].iloc[-1]
        
        # Avoid division by zero (e.g., on listing day glitches)
        if start_price <= 0: return None

        shares_bought = investment_amount / start_price 
        current_value = shares_bought * current_price
        
        # Calculate the series (Wealth over time)
        wealth_series = (hist_filtered['Close'] * shares_bought).round(0)

        return {
            'invested': investment_amount,
            'current_value': round(current_value, 0),
            'growth_pct': round(((current_value - investment_amount)/investment_amount)*100, 2),
            'start_price': round(start_price, 2),
            'curr_price': round(current_price, 2),
            # Format date nicely for display
            'start_date': hist_filtered.index[0].strftime('%b %Y'),
            'wealth_series': wealth_series.tolist(),
            # Format dates for the chart (YYYY-MM-DD is standard for charts)
            'dates': hist_filtered.index.strftime('%Y-%m-%d').tolist(),
            'is_adjusted': is_adjusted,
            'actual_start_year': actual_start_year
        }

    except Exception as e:
        # This prints exactly which stock failed and why
        print(f"Wealth Error for {getattr(ticker_obj, 'ticker', 'Unknown')}: {e}")
        return None

# ==========================================
# 5. DATA FETCHERS
# ==========================================

def get_detailed_financials(ticker_obj):
    statements = {}
    try:
        pl = ticker_obj.income_stmt
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        if pl.empty or bs.empty: return None
        
        years = [d.strftime('%Y') for d in pl.columns[:2]]
        statements['years'] = years

        # --- 1. INCOME STATEMENT ---
        statements['pl_rows'] = [
            get_accounting_row(pl, ['Total Revenue', 'Total Revenue'], 'Total Revenue', indent=0, bold=True),
            get_accounting_row(pl, ['Cost Of Revenue', 'Cost of Goods Sold'], 'Cost of Goods Sold (COGS)', indent=0, negative=True),
            # Gross Profit (Green)
            {'label': 'Gross Profit', 'values': [round(v/10000000, 2) for v in pl.loc['Gross Profit'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 700; color: #064e3b;'} if 'Gross Profit' in pl.index else None,
            
            {'label': 'Operating Expenses:', 'values': ['', ''], 'style': 'background-color: #f8fafc; font-weight: 700;'},
            get_accounting_row(pl, ['Selling General And Administration'], 'Selling, General & Admin (SG&A)', indent=1, negative=True),
            get_accounting_row(pl, ['Research And Development'], 'Research & Development (R&D)', indent=1, negative=True),
            
            # Operating Income (Green)
            {'label': 'Operating Income (or EBIT)', 'values': [round(v/10000000, 2) for v in pl.loc['Operating Income'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 700;'} if 'Operating Income' in pl.index else None,
            
            {'label': 'Other Income and (Expenses):', 'values': ['', ''], 'style': 'background-color: #f8fafc; font-weight: 700;'},
            get_accounting_row(pl, ['Interest Expense'], 'Interest Expense', indent=1, negative=True),
            
            get_accounting_row(pl, ['Pretax Income'], 'Income Before Tax', indent=0, bold=True),
            get_accounting_row(pl, ['Tax Provision'], 'Income Tax Expense', indent=0, negative=True),
            
            # Net Income (Green + Border)
            {'label': 'NET INCOME', 'values': [round(v/10000000, 2) for v in pl.loc['Net Income'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a; font-size: 1rem; color: #064e3b;'}
        ]
        statements['pl_rows'] = [x for x in statements['pl_rows'] if x]

        # --- 2. BALANCE SHEET ---
        statements['bs_rows'] = [
            {'label': 'ASSETS', 'values': ['', ''], 'style': 'font-weight: 800; text-transform: uppercase; background-color: #f1f5f9;'},
            {'label': 'Current Assets', 'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Cash And Cash Equivalents'], 'Cash and Cash Equivalents', indent=1),
            get_accounting_row(bs, ['Receivables'], 'Accounts Receivable', indent=1),
            get_accounting_row(bs, ['Inventory'], 'Inventory', indent=1),
            get_accounting_row(bs, ['Current Assets'], 'Total Current Assets', indent=0, bold=True),
            
            {'label': 'Non-Current Assets', 'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Net PPE'], 'Net Property, Plant, and Equipment', indent=1),
            get_accounting_row(bs, ['GoodwillAndIntangibleAssets'], 'Intangible Assets', indent=1),
            get_accounting_row(bs, ['Total Non Current Assets'], 'Total Non-Current Assets', indent=0, bold=True),
            
            # Total Assets (Green)
            {'label': 'TOTAL ASSETS', 'values': [round(v/10000000, 2) for v in bs.loc['Total Assets'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'},
            
            {'label': 'LIABILITIES AND STOCKHOLDERS\' EQUITY', 'values': ['', ''], 'style': 'font-weight: 800; text-transform: uppercase; background-color: #f1f5f9;'},
            {'label': 'Current Liabilities', 'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Accounts Payable'], 'Accounts Payable', indent=1),
            get_accounting_row(bs, ['Current Debt'], 'Current Portion of Debt', indent=1),
            get_accounting_row(bs, ['Current Liabilities'], 'Total Current Liabilities', indent=0, bold=True),
            
            {'label': 'Non-Current Liabilities', 'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Long Term Debt'], 'Long-Term Debt', indent=1),
            get_accounting_row(bs, ['Total Non Current Liabilities Net Minority Interest'], 'Total Non-Current Liabilities', indent=0, bold=True),
            
            # Total Liabilities (Red)
            {'label': 'Total Liabilities', 'values': [round(v/10000000, 2) for v in bs.loc['Total Liabilities Net Minority Interest'].values[:2]], 'style': 'background-color: #fee2e2; font-weight: 700; border-top: 2px solid #ef4444; color: #991b1b;'},
            
            {'label': 'Stockholders\' Equity', 'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Common Stock'], 'Common Stock', indent=1),
            get_accounting_row(bs, ['Retained Earnings'], 'Retained Earnings', indent=1),
            
            # Total Equity (Blue)
            {'label': 'Total Stockholders\' Equity', 'values': [round(v/10000000, 2) for v in bs.loc['Stockholders Equity'].values[:2]], 'style': 'background-color: #dbeafe; font-weight: 700;'},
            
            {'label': 'TOTAL LIABILITIES AND EQUITY', 'values': [round(v/10000000, 2) for v in bs.loc['Total Assets'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'}
        ]

        # --- 3. CASH FLOW ---
        statements['cf_rows'] = [
            {'label': 'Cash Flow from Operating Activities', 'values': ['', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(cf, ['Net Income', 'Net Income From Continuing Operations'], 'Net Income', indent=1),
            get_accounting_row(cf, ['Depreciation'], 'Add: Depreciation', indent=1),
            get_accounting_row(cf, ['Operating Cash Flow'], 'Net Cash from Operating Activities', indent=0, bold=True),
            
            {'label': 'Cash Flow from Investing Activities', 'values': ['', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(cf, ['Capital Expenditure'], 'Purchase of PPE', indent=1, negative=True),
            get_accounting_row(cf, ['Investing Cash Flow'], 'Net Cash used in Investing Activities', indent=0, bold=True),
            
            {'label': 'Cash Flow from Financing Activities', 'values': ['', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(cf, ['Cash Dividends Paid'], 'Payment of dividends', indent=1, negative=True),
            get_accounting_row(cf, ['Financing Cash Flow'], 'Net Cash from Financing Activities', indent=0, bold=True),
            
            # Net Change (Blue)
            {'label': 'Net Change in Cash', 'values': [round(v/10000000, 2) for v in cf.loc['End Cash Position'].values[:2]], 'style': 'background-color: #dbeafe; font-weight: 800; border-top: 2px solid #2563eb;'} if 'End Cash Position' in cf.index else None
        ]
        statements['cf_rows'] = [x for x in statements['cf_rows'] if x]

        # --- 4. RETAINED EARNINGS ---
        re_vals = bs.loc['Retained Earnings'].values[:2] if 'Retained Earnings' in bs.index else [0, 0]
        re_list = [round(v/10000000, 2) for v in re_vals]
        while len(re_list) < 2: re_list.append(0)
        
        statements['re_rows'] = [
            {'label': 'Beginning Retained Earnings', 'values': [re_list[1]], 'style': 'font-weight: 700;'},
            {'label': 'Plus: Net Income', 'values': [round(pl.loc['Net Income'].iloc[0]/10000000, 2) if 'Net Income' in pl.index else 0], 'style': 'padding-left: 20px;'},
            {'label': 'Less: Dividends Paid', 'values': [round(cf.loc['Cash Dividends Paid'].iloc[0]/10000000, 2) if 'Cash Dividends Paid' in cf.index else 0], 'style': 'padding-left: 20px; color: #991b1b;'},
            
            # ENDING RE (Green)
            {'label': 'Ending Retained Earnings', 'values': [re_list[0]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'}
        ]
    except: return None
    return statements


def get_peer_data_with_share(ticker, sector, current_mcap):
    # 1. Resolve Table Path
    client = get_bq_client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    table_id = f"{client.project}.{dataset_id}.stock_intelligence_v3"
    
    peers_list = []
    
    # 2. Get UNIQUE tickers currently assigned to this sector
    # We use ROW_NUMBER to only look at the MOST RECENT record for every stock.
    # This prevents ITC (now FMCG) from appearing in INFRASTRUCTURE results.
    try:
        sql = f"""
            SELECT ticker FROM (
                SELECT ticker, sector, 
                       ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY last_updated DESC) as rn
                FROM `{table_id}`
            )
            WHERE rn = 1 
            AND sector = '{sector}' 
            AND ticker != '{ticker}'
            LIMIT 5
        """
        df = client.query(sql).to_dataframe()
        peers_list = df['ticker'].tolist()
    except Exception as e:
        print(f"⚠️ Peer SQL Error: {e}")

    # Fallback if DB is empty or query fails
    if not peers_list: 
        if "TECH" in sector: peers_list = ["TCS", "INFY", "HCLTECH"]
        elif "BANK" in sector: peers_list = ["HDFCBANK", "ICICIBANK", "SBIN"]
        elif "FMCG" in sector: peers_list = ["HINDUNILVR", "NESTLEIND", "DABUR"]
        else: peers_list = []
        
    peer_data = []
    total_sector_mcap = current_mcap
    
    # 3. Fetch Live Data for the Peers
    for p in peers_list: 
        try:
            if not p or len(p) < 2: continue

            # Handle suffix logic
            ticker_suffix = p if (".NS" in p or ".BO" in p) else f"{p}.NS"
            p_obj = yf.Ticker(ticker_suffix)
            p_info = p_obj.info
            
            if not p_info or 'currentPrice' not in p_info:
                continue

            mcap = to_crores(p_info.get('marketCap', 0))
            
            if mcap > 0:
                total_sector_mcap += mcap
                peer_data.append({
                    'ticker': p,
                    'name': p_info.get('shortName', p),
                    'price': p_info.get('currentPrice', 0),
                    'mcap': mcap,
                    'pe': round(safe_float(p_info.get('trailingPE') or p_info.get('forwardPE') or 0), 1),
                    'roe': round(safe_float(p_info.get('returnOnEquity') or 0) * 100, 1)
                })
        except Exception:
            continue

    # 4. Calculate Market Share percentages
    for p in peer_data: 
        p['share'] = round((p['mcap'] / total_sector_mcap) * 100, 1) if total_sector_mcap > 0 else 0
    
    current_share = round((current_mcap / total_sector_mcap) * 100, 1) if total_sector_mcap > 0 else 100
    
    # Return sorted by Market Cap
    return sorted(peer_data, key=lambda x: x['mcap'], reverse=True), current_share

def get_dynamic_peers(ticker, sector):
    client = get_bq_client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    table_id = f"{client.project}.{dataset_id}.stock_intelligence"
    
    try:
        # Standard Query: Get top 5 largest companies in the SAME sector
        query = f"""
            SELECT ticker 
            FROM `{table_id}` 
            WHERE sector = '{sector}' 
            AND ticker != '{ticker}' 
            GROUP BY ticker 
            ORDER BY MAX(mcap) DESC 
            LIMIT 5
        """
        
        df = client.query(query).to_dataframe()
        
        if not df.empty:
            return df['ticker'].tolist()
        else:
            return []

    except Exception as e:
        print(f"Error fetching peers: {e}")
        return []

def get_safe_shareholding(ticker_obj, stock_info):
    # 1. Debug Printing
    if isinstance(stock_info, dict):
        print(f"🕵️ DEBUG SHAREHOLDING KEYS: {list(stock_info.keys())}")
        print(f"Insiders: {stock_info.get('heldPercentInsiders')}, Institutions: {stock_info.get('heldPercentInstitutions')}")
    
    data = { "promoter": 0, "institution": 0, "public": 0, "top_investors": [] }
    
    try:
        # --- SAFETY CHECK ---
        if not isinstance(stock_info, dict):
            stock_info = {}

        # Get total shares to help convert raw numbers to percentages if needed
        total_shares = stock_info.get('sharesOutstanding') or 1

        # --- STRATEGY 1: Check .info (Fastest) ---
        p_hold = stock_info.get('heldPercentInsiders')
        i_hold = stock_info.get('heldPercentInstitutions')
        
        if p_hold is not None: data['promoter'] = round(p_hold * 100, 2)
        if i_hold is not None: data['institution'] = round(i_hold * 100, 2)
            
        # --- STRATEGY 2: Scrape Major Holders (If info is empty) ---
        if data['promoter'] == 0:
            try:
                major = ticker_obj.major_holders
                if major is not None and not major.empty:
                    for idx, row in major.iterrows():
                        row_str = str(row.values).lower() 
                        val = 0
                        for cell in row:
                            if isinstance(cell, str) and '%' in cell:
                                try:
                                    val = float(cell.replace('%', '').strip())
                                    break
                                except: continue
                        if "insider" in row_str:
                            data['promoter'] = val
                        elif "institut" in row_str:
                            data['institution'] = val
            except: pass

        # --- STRATEGY 3: Top Investors (The Smartest Version) ---
        try:
            # Try Institutional Holders first
            inst = ticker_obj.institutional_holders
            # FALLBACK: Try Mutual Funds if Institutional is empty
            if inst is None or inst.empty:
                inst = ticker_obj.mutualfund_holders
            
            if inst is not None and not inst.empty:
                inst.columns = [str(c).lower() for c in inst.columns]
                
                for idx, row in inst.head(5).iterrows():
                    # Find Name
                    name = row.get('holder', row.iloc[0] if len(row) > 0 else 'Unknown')
                    
                    # Find Stake % (Aggressive detection)
                    stake_val = 0
                    for cell in row:
                        try:
                            val = float(str(cell).replace('%', '').strip())
                            if val <= 0: continue
                            
                            if val > 100: # It's a share count, convert it
                                stake_val = (val / total_shares) * 100
                            elif val < 1: # It's a decimal ratio
                                stake_val = val * 100
                            else: # It's a standard percentage
                                stake_val = val
                            if stake_val > 0: break
                        except: continue
                    
                    if stake_val > 0:
                        # Return as a NUMBER for the template to handle
                        data['top_investors'].append({
                            'name': name, 
                            'stake': round(stake_val, 2)
                        })
        except: pass

        # --- 4. Final Public Calculation ---
        total_known = data['promoter'] + data['institution']
        if 0 < total_known < 100:
            data['public'] = round(100 - total_known, 2)

    except Exception as e:
        print(f"❌ Critical Shareholding Error: {e}")

    return data




# ==========================================
# 6. SYNC & MAIN CONTROLLER
# ==========================================


def sync_stock_on_demand(query):
    # --- STEP 1: RESOLVE SYMBOL (Smart Search) ---
    try:
        symbol = resolve_symbol_from_name(query)
    except NameError:
        symbol = query.upper().replace(".", "").replace("LTD", "").strip()

    # =========================================================
    # 2. SPEED LAYER: Check Local RAM Cache (0.001s)
    # =========================================================
    safe_key = slugify(symbol) 
    cache_key = f"stock_data_{safe_key}"

    cached_payload = cache.get(cache_key)
    if cached_payload:
        return cached_payload

    # Setup BigQuery Client
    client = None
    table_id = None
    try:
        client = get_bq_client()
        dataset_id = os.getenv("GCP_DATASET_ID")
        table_id = f"{client.project}.{dataset_id}.stock_intelligence"
    except Exception as e:
        print(f"⚠️ BigQuery Client Init Failed: {e}")

    # =========================================================
    # 3. WAREHOUSE LAYER: Check BigQuery (0.5s - 1s)
    # =========================================================
    if client and table_id:
        try:
            # Check for data less than 60 minutes old
            sql = f"""
                SELECT * FROM `{table_id}` 
                WHERE ticker = '{symbol}' 
                # This tells the code: "Ignore everything, fetch NEW data!"
                AND last_updated > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 60 MINUTE)
                ORDER BY last_updated DESC
                LIMIT 1
            """
            bq_data = client.query(sql).to_dataframe()
            
            if not bq_data.empty:
                print(f"⚡ BIGQUERY HIT: Serving {symbol} from Cloud")
                row = bq_data.iloc[0].to_dict()
                
                # Deserialize JSON strings back into Python Lists/Dicts
                # We include 'shareholding' here to read it back correctly
                json_cols = ['technicals', 'policy', 'arth_score', 'news', 'chart_dates', 'chart_prices', 'chart_volumes', 'shareholding']
                try:
                    for field in json_cols:
                        if isinstance(row.get(field), str): 
                            row[field] = json.loads(row[field])
                except: pass

                cache.set(cache_key, row, timeout=60*15)
                return row
        except Exception as e:
            print(f"⚠️ BigQuery Read Failed (Continuing...): {e}")

    # =========================================================
    # 4. LIVE LAYER: Fetch Fresh Data (2s - 4s)
    # =========================================================
    print(f"🔄 LIVE FETCH: Getting fresh data for {symbol}...")
    
    try:
        # --- SMART TICKER LOGIC ---
        if ".BO" in symbol:
            ticker_name = symbol
            clean_symbol = symbol 
        else:
            ticker_name = f"{symbol}.NS"
            clean_symbol = symbol 
            
        ticker_obj = yf.Ticker(ticker_name)
        
        # --- CRITICAL FIX: Rename 'info' to 'stock_info' ---
        try: stock_info = ticker_obj.info
        except: return None 

        if not stock_info or 'currentPrice' not in stock_info or stock_info['currentPrice'] is None: 
            return None

        # --- RAW METRICS ---
        hist = ticker_obj.history(period="5y")
        
        # Safe calculations
        try: roe = round(safe_float(stock_info.get('returnOnEquity')) * 100, 2)
        except: roe = 0
        
        raw_de = safe_float(stock_info.get('debtToEquity'))
        debt_to_equity = round(raw_de / 100, 2) if raw_de > 5 else round(raw_de, 2)
        
        pe_val = stock_info.get('trailingPE') or stock_info.get('forwardPE') or 0
        
        # CAGR Calculation
        cagr_val = 0
        if not hist.empty:
            start_p = hist['Close'].iloc[0]
            end_p = hist['Close'].iloc[-1]
            if start_p > 0: cagr_val = round(((end_p / start_p) ** (1 / 5) - 1) * 100, 2)

        # --- INTELLIGENCE MODULES ---
        raw_sector = stock_info.get('sector', 'Unknown')
        raw_industry = stock_info.get('industry', 'Unknown')
        description = stock_info.get('longBusinessSummary', '')
        
        # REFINED SECTOR: Passes Description + Clean Symbol (Fixes Reliance/Adani)
        refined_sector = refine_sector_name(raw_sector, raw_industry, clean_symbol, description)
        
        tech_signals = calculate_technicals(hist)
        news_data = get_robust_news(ticker_obj, symbol)
        policy_context = get_policy_context(refined_sector, str(raw_industry).upper())
        
        # SHAREHOLDING: Connects to the new safe function
        shareholding_pattern = get_safe_shareholding(ticker_obj, stock_info)

        arth_analysis = calculate_arth_score(
            financials={'roe': roe, 'profit_growth': round(safe_float(stock_info.get('earningsGrowth')) * 100, 2), 'debt_to_equity': debt_to_equity},
            technicals=tech_signals,
            policy=policy_context,
            news_list=news_data
        )

        # --- DATA PAYLOAD ---
        data_payload = {
            'ticker': symbol,
            'company_name': str(stock_info.get('longName', symbol)),
            'sector': refined_sector,
            'price': safe_float(stock_info.get('currentPrice')),
            'day_high': safe_float(stock_info.get('dayHigh')),
            'day_low': safe_float(stock_info.get('dayLow')),
            '52_high': safe_float(stock_info.get('fiftyTwoWeekHigh')),
            '52_low': safe_float(stock_info.get('fiftyTwoWeekLow')),
            'mcap': to_crores(stock_info.get('marketCap')),
            'enterprise_value': to_crores(stock_info.get('enterpriseValue')),
            'shares_fmt': format_shares(stock_info.get('sharesOutstanding')),
            'pb_ratio': safe_float(stock_info.get('priceToBook')),
            'div_yield': round(safe_float(stock_info.get('dividendYield')) * 100, 2),
            'cash': to_crores(stock_info.get('totalCash')),
            'debt': to_crores(stock_info.get('totalDebt')),
            'promoter_holding': round(safe_float(stock_info.get('heldPercentInsiders')) * 100, 2),
            'roe': roe,
            'profit_growth': round(safe_float(stock_info.get('earningsGrowth')) * 100, 2),
            'pe_ratio': round(safe_float(pe_val), 2),
            'industry_pe': get_industry_pe(refined_sector, symbol, FALLBACK_SECTOR_PE.get(refined_sector, 20.0)),
            'debt_to_equity': debt_to_equity,
            'cagr_5y': cagr_val,
            'technicals': tech_signals, 
            'news': news_data, 
            'policy': policy_context,
            'arth_score': arth_analysis,
            'shareholding': shareholding_pattern,
            'last_updated': datetime.now(pytz.timezone('Asia/Kolkata'))
        }

        # Handle Chart Data
        if not hist.empty:
            data_payload['chart_dates'] = hist.index.strftime('%Y-%m-%d').tolist()
            data_payload['chart_prices'] = hist['Close'].round(2).tolist()
            data_payload['chart_volumes'] = hist['Volume'].fillna(0).tolist()
        else:
            data_payload['chart_dates'] = []
            data_payload['chart_prices'] = []
            data_payload['chart_volumes'] = []

        # =========================================================
        # 5. SAVE LAYER: Save to Cache & BigQuery
        # =========================================================
        
        # A. Save to RAM
        cache.set(cache_key, data_payload, timeout=60*15)

        # B. Save to BigQuery (Storage)
        if client and table_id:
            try:
                # 1. CHANGE TABLE NAME to '_v3' to force a fresh start
                # This fixes the "Schema Mismatch" error instantly.
                if "_v2" in table_id:
                    table_id = table_id.replace("_v2", "_v3")
                elif "_v3" not in table_id:
                    table_id = table_id + "_v3"

                # 2. Prepare Record
                bq_record = data_payload.copy()
                bq_record['last_updated'] = datetime.now(pytz.UTC)

                # 3. CONVERT DICTS TO JSON STRINGS (Crucial)
                json_fields = [
                    'technicals', 'news', 'policy', 'arth_score', 
                    'chart_dates', 'chart_prices', 'chart_volumes', 
                    'shareholding'
                ]
                
                for field in json_fields:
                    val = bq_record.get(field)
                    # Force conversion to String, handling None/Empty safely
                    if val is None:
                        bq_record[field] = "{}" # Empty JSON object
                    elif isinstance(val, (dict, list)):
                        bq_record[field] = json.dumps(val, default=str)
                    else:
                        bq_record[field] = str(val)

                # 4. Create DataFrame
                df = pd.DataFrame([bq_record])
                
                # 5. FORCE PANDAS TYPES (The "Pyarrow" Fix)
                # We explicitly tell Pandas: "These columns are STRINGS, not Objects"
                for col in json_fields:
                    if col in df.columns:
                        df[col] = df[col].astype("string") # Use specific 'string' type

                # 6. Upload
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND", 
                    schema_update_options=["ALLOW_FIELD_ADDITION"]
                )
                client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
                print(f"✅ Data Saved to BigQuery Table: {table_id}")
                
            except Exception as e:
                print(f"⚠️ BigQuery Upload Skipped: {e}")

        return data_payload
    
    except Exception as e:
        print(f"❌ Critical Sync Error: {e}")
        return None



def index(request):
    # 1. Get User Input
    raw_query = request.GET.get('q', 'RELIANCE').upper().strip()
    wealth_amount = int(request.GET.get('w_amt', 100000))
    wealth_year = int(request.GET.get('w_year', 2011))

    # 2. Fetch Main Data (This includes Shareholding!)
    data = sync_stock_on_demand(raw_query)
    
    # Handle Invalid Stock
    if not data: 
        return render(request, 'dashboard/index.html', {'error_message': f"Ticker '{raw_query}' not valid."})

    symbol = data['ticker']

    # 3. Re-create Ticker Object (Needed for Financials & Wealth functions only)
    # We must handle the suffix correctly again here
    if ".BO" in symbol:
        t_name = symbol
    else:
        t_name = f"{symbol}.NS"
    
    ticker_obj = yf.Ticker(t_name)
    
    # 4. Run Analysis Modules (Using ticker_obj)
    try:
        financials = get_detailed_financials(ticker_obj)
        fin_analysis = analyze_financial_health(ticker_obj)
        wealth_data = calculate_wealth_growth(ticker_obj, wealth_amount, wealth_year)
    except Exception as e:
        print(f"⚠️ Analysis Module Error: {e}")
        financials, fin_analysis, wealth_data = {}, {}, None

    # 5. GET SHAREHOLDING (No new calculation needed!)
    # We just grab it from the 'data' payload we already fetched
    shareholding = data.get('shareholding', {})
    
    # 6. Prepare Pie Chart Data manually
    pie_labels_list = ["Promoters", "Institutions", "Public & Others"]
    pie_data_list = [
        shareholding.get('promoter', 0), 
        shareholding.get('institution', 0), 
        shareholding.get('public', 0)
    ]

    # 7. Benchmark Comparison (Nifty vs Stock)
    bench_sym = SECTOR_BENCHMARKS.get(data['sector'], "^NSEI")
    bench_stats = {}
    dates, stock_pct, bench_pct, volumes = [], [], [], []

    try:
        b_hist = yf.Ticker(bench_sym).history(period="5y")
        
        # Ensure we have chart data before comparing
        if 'chart_prices' in data and data['chart_prices']:
            min_len = min(len(data['chart_prices']), len(b_hist))
            
            stock_series = data['chart_prices'][-min_len:]
            bench_vals = b_hist['Close'].values[-min_len:].tolist()
            dates = data['chart_dates'][-min_len:]
            volumes = data['chart_volumes'][-min_len:]
            
            if bench_vals and stock_series:
                b_start, b_end = bench_vals[0], bench_vals[-1]
                s_start, s_end = stock_series[0], stock_series[-1]
                
                bench_stats = {
                    'symbol': bench_sym.replace('^', ''),
                    'return_pct': round(((b_end - b_start)/b_start)*100, 2) if b_start != 0 else 0,
                    'stock_return_pct': round(((s_end - s_start)/s_start)*100, 2) if s_start != 0 else 0
                }
                
                stock_pct = [round(((p - s_start)/s_start)*100, 2) for p in stock_series]
                bench_pct = [round(((p - b_start)/b_start)*100, 2) for p in bench_vals]
    except Exception as e:
        print(f"⚠️ Benchmark Error: {e}")

    # 8. Prepare Context
    context = {
        'symbol': symbol,
        'data': data,
        'narendra_rating': calculate_narendra_rating(data),
        # Using safe index access for peers in case list is empty
        'peers': get_peer_data_with_share(symbol, data['sector'], data['mcap'])[0] if get_peer_data_with_share(symbol, data['sector'], data['mcap']) else [],
        'my_share': get_peer_data_with_share(symbol, data['sector'], data['mcap'])[1] if get_peer_data_with_share(symbol, data['sector'], data['mcap']) else {},
        
        'financials': financials,
        'fin_analysis': fin_analysis,
        'shareholding': shareholding,
        
        # Chart JSON Data
        'pie_labels': json.dumps(pie_labels_list),
        'pie_data': json.dumps(pie_data_list),
        'chart_dates': json.dumps(dates),
        'chart_stock_pct': json.dumps(stock_pct),
        'chart_bench_pct': json.dumps(bench_pct),
        'chart_volumes': json.dumps(volumes),
        
        'bench_stats': bench_stats,
        'wealth_data': wealth_data,
        'wealth_series': json.dumps(wealth_data['wealth_series']) if wealth_data else "[]",
        'wealth_dates': json.dumps(wealth_data['dates']) if wealth_data else "[]",
        'w_amt': wealth_amount,
        'w_year': wealth_year,
        'year_range': list(range(2000, datetime.now().year + 1)),
        'cagr': data.get('cagr_5y', 0),
        'bench_label': bench_sym.replace("^", "").replace("CNX", "NIFTY ")
    }
    return render(request, 'dashboard/index.html', context)