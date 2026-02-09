import os
import json
import sys
import pytz
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

def resolve_symbol_from_name(query):
    # 1. CLEAN THE INPUT
    clean_query = query.upper()
    for word in [".", " LTD", " LIMITED", " INDIA", " PVT", " PRIVATE", " INC"]:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()

    # 2. CHECK MANUAL ALIASES
    if clean_query in CUSTOM_ALIASES:
        return CUSTOM_ALIASES[clean_query]

    # 3. CHECK CACHE (Safe Check)
    # We DO NOT download if missing. This prevents the startup crash.
    full_mapping = cache.get('nse_master_mapping')
    
    if full_mapping:
        if clean_query in full_mapping:
            return full_mapping[clean_query]
        # Partial match logic
        for company_name, ticker in full_mapping.items():
            if clean_query in company_name.replace("LIMITED", "").strip():
                return ticker

    # 4. Fail-Safe: Just return the query itself
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

    # --- 1. CLEAN THE TICKER ---
    sym = str(ticker).upper().replace(".NS", "").replace(".BO", "").strip()

    # DEBUG
    print(f"🕵️ SECTOR CHECK: Input='{ticker}' -> Cleaned='{sym}'")

    desc = str(description).upper() if description else ""

    # =========================================================
    # 2. GOD MODE: HARDCODED OVERRIDES (Highest Priority)
    # =========================================================

    # FMCG
    if sym in ["ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "MARICO", "VBL", "VARUN BEVERAGES", "COLPAL"]:
        print(f"✅ God Mode Triggered: {sym} -> FMCG")
        return "FMCG"
    
    # AUTOMOBILE
    if sym in ["MARUTI", "TATAMOTORS", "M&M", "ASHOKLEY", "HEROMOTOCO", "EICHERMOT", "BAJAJ-AUTO", "TVSMOTOR"]:
        return "AUTOMOBILE"

    # NBFC (Added PFC, REC, IRFC here because they often get confused with Power/Railways)
    if sym in ["SHRIRAMFIN", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN", "JIOFIN", "PFC", "REC", "IRFC", "M&MFIN"]:
        return "NBFC" 

    # PHARMA
    if sym in ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "APOLLOHOSP", "TORNTPHARM", "ALKEM"]:
        return "PHARMA"

    # METALS & MINING (New)
    if sym in ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC", "SAIL", "HINDZINC"]:
        return "METALS"

    # CEMENT (New)
    if sym in ["ULTRACEMCO", "AMBUJACEM", "ACC", "SHREECEM", "DALBHARAT", "RAMCOCEM"]:
        return "CEMENT"

    # BANKING (Safety Net)
    if sym in ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB"]:
        return "BANKING"

    # Oil & Gas
    if sym in ["RELIANCE", "ONGC", "OIL", "IOC", "BPCL", "HPCL", "GAIL", "IGL", "MGL"]:
        return "OIL & GAS"

    # Aviation
    if sym in ["INDIGO", "SPICEJET", "JETAIRWAYS"]:
        return "AVIATION"

    # Infrastructure & Logistics
    if sym in ["ADANIPORTS", "GPPL", "JSWINFRA", "IRB", "GMRINFRA", "L&T", "LT", "RVNL", "IRCON"]:
        return "INFRASTRUCTURE"

    if sym in ["CONCOR", "VRL", "TCI", "BLUE DART", "BLUEDART"]:
        return "LOGISTICS"

    # Power & Renewables
    if sym in ["SUZLON", "INOXWIND", "KPIGREEN", "ADANIGREEN", "TATAPOWER", "NTPC", "POWERGRID", "SJVN", "NHPC", "ADANIPOWER"]:
        return "POWER & RENEWABLES"

    # Consumer Durables
    if sym in ["BLUESTARCO", "VOLTAS", "WHIRLPOOL", "CROMPTON", "HAVELLS", "POLYCAB", "DIXON", "AMBER", "KAJARIA", "KEI"]:
        return "CONSUMER DURABLES"

    # Capital Goods
    if sym in ["HONAUT", "ABB", "SIEMENS", "THERMAX", "CUMMINSIND", "BHEL", "BEL", "HAL"]:
        return "CAPITAL GOODS"

    # Consumer Tech
    if sym in ["ZOMATO", "PAYTM", "NYKAA", "POLICYBZR", "DELHIVERY", "NAUKRI", "CARTRADE"]:
        return "CONSUMER TECH"

    # IT Services
    if sym in ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "KPITTECH"]:
        return "IT SERVICES"

    # =========================================================
    # 3. DYNAMIC SCANNING (Keyword Search)
    # =========================================================

    if "AIRLINE" in desc or "AVIATION" in desc: return "AVIATION"
    if "CEMENT" in desc or "CONCRETE" in desc: return "CEMENT"
    if "STEEL" in desc or "ALUMINUM" in desc or "COPPER" in desc or "MINING" in desc: return "METALS"
    
    # Infrastructure keywords
    if "PORTS" in desc or "HIGHWAY" in desc or "TOLL" in desc or "EPC" in desc or "CONSTRUCTION PROJECT" in desc:
        return "INFRASTRUCTURE"

    if "LOGISTICS" in desc or "CARGO" in desc or "FREIGHT" in desc or "WAREHOUS" in desc or "TRANSPORT" in desc:
        return "LOGISTICS"

    if "WIND ENERGY" in desc or "SOLAR" in desc or "HYDRO" in desc or "THERMAL POWER" in desc:
        return "POWER & RENEWABLES"

    if "AIR CONDITION" in desc or "REFRIGERATOR" in desc or "CABLES" in desc or "TILES" in desc:
        return "CONSUMER DURABLES"

    if "DEFENSE" in desc or "MISSILE" in desc or "NAVAL" in desc or "AEROSPACE" in desc:
        return "DEFENSE"

    # =========================================================
    # 4. FALLBACK
    # =========================================================
    if "BANK" in ind: return "BANKING"
    if "TECHNOLOGY" in sec: return "IT SERVICES"
    if "PHARMA" in ind: return "PHARMA"
    if "AUTO" in ind: return "AUTOMOBILE"
    if "REAL ESTATE" in sec: return "REALTY"
    if "FMCG" in sec or "FOOD" in ind: return "FMCG"
    if "BASIC MATERIALS" in sec: return "METALS" # Good fallback for metals

    return sec

def get_bq_client():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    key_path = BASE_DIR / "service-account.json"
    credentials = service_account.Credentials.from_service_account_file(
        str(key_path))
    return bigquery.Client(credentials=credentials, project=credentials.project_id)


def to_crores(value):
    try:
        return round(float(value) / 10000000, 2) if value else 0
    except:
        return 0


def format_shares(value):
    try:
        val = float(value)
        if val > 1_000_000_000:
            return f"{round(val / 1_000_000_000, 2)} B"
        if val > 10_000_000:
            return f"{round(val / 10_000_000, 2)} Cr"
        return str(value)
    except:
        return "N/A"


def safe_float(val):
    try:
        if hasattr(val, 'iloc'):
            return float(val.iloc[0])
        if hasattr(val, 'item'):
            return float(val.item())
        if val is None or pd.isna(val):
            return 0.0
        return float(val)
    except:
        return 0.0


def calculate_cagr(start_price, end_price, years):
    if start_price == 0 or years == 0:
        return 0.0
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
            if val > 0:
                return round(val, 2)
    except:
        pass
    return fallback_val


def get_accounting_row(df, keys, label, indent=0, bg_color=None, bold=False, negative=False):
    style_str = f"padding-left: {indent * 20 + 10}px;"
    if bg_color:
        style_str += f" background-color: {bg_color};"
    if bold:
        style_str += " font-weight: 700;"

    vals = [0, 0]
    for k in keys:
        if k in df.index:
            raw_vals = df.loc[k].values[:2]
            fmt_vals = []
            for v in raw_vals:
                val_cr = round(v / 10000000, 2)
                if negative and val_cr > 0:
                    val_cr = -val_cr
                fmt_vals.append(val_cr)
            while len(fmt_vals) < 2:
                fmt_vals.append(0)
            vals = fmt_vals
            break
    return {'label': label, 'values': vals, 'style': style_str}

# --- NEW: Technical Analysis Engine ---


def calculate_technicals(hist):
    """
    Calculates key technical indicators from historical data.
    """
    if hist.empty:
        return None

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
    if rsi > 70:
        signals['rsi'] = {'val': rsi,
                          'status': 'Overbought', 'color': 'text-danger'}
    elif rsi < 30:
        signals['rsi'] = {'val': rsi,
                          'status': 'Oversold', 'color': 'text-success'}
    else:
        signals['rsi'] = {'val': rsi,
                          'status': 'Neutral', 'color': 'text-warning'}

    # SMA Trend
    price = latest['Close']
    sma50 = latest['SMA_50']
    sma200 = latest['SMA_200']

    if price > sma50 and price > sma200:
        signals['trend'] = {'status': 'Strong Bullish',
                            'color': 'text-success', 'desc': 'Price > 50 & 200 DMA'}
    elif price < sma50 and price < sma200:
        signals['trend'] = {'status': 'Strong Bearish',
                            'color': 'text-danger', 'desc': 'Price < 50 & 200 DMA'}
    elif price > sma200:
        signals['trend'] = {'status': 'Bullish Bias',
                            'color': 'text-primary', 'desc': 'Price > 200 DMA'}
    else:
        signals['trend'] = {'status': 'Bearish Bias',
                            'color': 'text-warning', 'desc': 'Price < 200 DMA'}

    # MACD Signal
    macd = latest['MACD']
    sig = latest['Signal_Line']
    if macd > sig:
        signals['macd'] = {'val': round(
            macd, 2), 'status': 'Bullish Crossover', 'color': 'text-success'}
    else:
        signals['macd'] = {'val': round(
            macd, 2), 'status': 'Bearish Crossover', 'color': 'text-danger'}

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
                if not title:
                    continue
                ts = n.get('providerPublishTime') or n.get('pubDate') or 0
                date_str = datetime.fromtimestamp(ts).strftime(
                    '%d %b %Y') if ts else "Recent"
                news_data.append({
                    'title': title,
                    'link': n.get('link') or n.get('url'),
                    'publisher': n.get('publisher') or n.get('provider') or "Yahoo Finance",
                    'date': date_str,
                    'source': 'Yahoo'
                })
    except:
        pass

    # 2. Fallback to Google News RSS
    if not news_data:
        try:
            query = f"{symbol} stock news India"
            encoded_query = query.replace(" ", "%20")
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                try:
                    dt = datetime.strptime(
                        entry.published[:16], '%a, %d %b %Y')
                    date_str = dt.strftime('%d %b %Y')
                except:
                    date_str = "Recent"
                news_data.append({
                    'title': entry.title,
                    'link': entry.link,
                    'publisher': entry.source.title if hasattr(entry, 'source') else "Google News",
                    'date': date_str,
                    'source': 'Google'
                })
        except Exception as e:
            print(f"Google News Error: {e}")
    return news_data

# --- POLICY MAPPER ---
def get_policy_context(sector_name, industry_name):
    # 1. LOAD THE JSON FILE
    policy_data = None
    try:
        # Robust path finding for Django/Render/Docker environments
        base_dir = Path(__file__).resolve().parent.parent.parent # Goes up to 'src'
        
        # Try primary location
        file_path = base_dir / 'data' / 'policy_context.json'
        
        # Try fallback location (if src is root)
        if not file_path.exists():
            file_path = base_dir / 'src' / 'data' / 'policy_context.json'

        with open(file_path, 'r') as f:
            policy_data = json.load(f)
            
    except Exception as e:
        print(f"⚠️ Policy File Error: {e}")
        return None

    if not policy_data: 
        return None

    # 2. MAP SECTOR TO JSON KEY
    sec = str(sector_name).upper()
    ind = str(industry_name).upper()
    target_key = "INFRASTRUCTURE" # Default fallback

    # Exact mappings to matches your JSON keys
    if "TEXTILE" in sec or "APPAREL" in ind: target_key = "TEXTILES"
    elif "AGRI" in sec or "FARM" in ind: target_key = "AGRICULTURE"
    elif "RAIL" in ind: target_key = "RAILWAYS"
    elif "DEFENCE" in ind or "AEROSPACE" in ind: target_key = "DEFENCE"
    elif "CHEM" in sec or "CHEM" in ind: target_key = "CHEMICALS"
    elif "AUTO" in sec or "VEHICLE" in ind: target_key = "AUTO"
    elif "BANK" in sec: target_key = "BANKING"
    elif "IT" in sec or "TECH" in sec: target_key = "IT"
    elif "PHARMA" in sec or "DRUG" in ind: target_key = "PHARMA"
    elif "FMCG" in sec or "FOOD" in ind: target_key = "FMCG"
    elif "POWER" in sec or "ENERGY" in sec or "WIND" in ind or "SOLAR" in ind: target_key = "POWER"
    elif "REAL" in sec or "REIT" in ind: target_key = "REALTY"
    elif "NBFC" in sec or "FINANCE" in sec: target_key = "NBFC"
    # Catch-all for Infra
    elif "INFRA" in sec or "CONSTRUCT" in ind or "PORT" in ind or "ROAD" in ind: target_key = "INFRASTRUCTURE"

    # 3. RETURN DATA
    if target_key in policy_data['sectors']:
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
        if bs.empty or cf.empty or pl.empty:
            return []

        latest_cfo = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
        latest_ni = pl.loc['Net Income'].iloc[0] if 'Net Income' in pl.index else 0
        if latest_cfo > latest_ni:
            analysis.append({"title": "High Quality Earnings", "status": "Good",
                            "desc": "Operating Cash Flow > Net Income."})
        else:
            analysis.append({"title": "Aggressive Accounting?",
                            "status": "Caution", "desc": "Net Income > Cash Flow."})

        total_debt = bs.loc['Total Debt'].iloc[0] if 'Total Debt' in bs.index else 0
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        if cash > total_debt:
            analysis.append({"title": "Balance Sheet Fortress",
                            "status": "Good", "desc": "Cash > Total Debt."})
        else:
            analysis.append({"title": "Leveraged Balance Sheet", "status": "Neutral",
                            "desc": f"Net Debt: {to_crores(total_debt - cash)} Cr."})
    except:
        pass
    return analysis


# --- SCORING ENGINE FUNCTION ---
# ==========================================
# REPLACE YOUR EXISTING calculate_arth_score WITH THIS
# ==========================================
def calculate_arth_score(financials, technicals, policy, news_list, sector):
    """
    The Brain: Calculates a 0-100 Health Score & Recommendation.
    """
    score = 0
    
    # 1. FINANCIALS (Max 40 Points)
    roe = financials.get('roe', 0)
    profit_growth = financials.get('profit_growth', 0)
    debt_equity = financials.get('debt_to_equity', 0)

    # ROE (Return on Equity)
    if roe > 20: score += 15
    elif roe > 15: score += 10
    elif roe > 10: score += 5

    # Profit Growth
    if profit_growth > 20: score += 15
    elif profit_growth > 10: score += 10
    elif profit_growth > 0: score += 5

    # Debt Logic (SMART FIX FOR BANKS/POWER)
    # Banks & Power Cos (like Suzlon) have high debt naturally.
    # We ignore debt penalty if ROE is good.
    if "BANK" in sector or "FINANCE" in sector or "POWER" in sector or "INFRA" in sector:
        if roe > 10: score += 10
    else:
        # Normal Companies: Penalize High Debt
        if debt_equity < 0.5: score += 10
        elif debt_equity < 1.0: score += 5

    # 2. TECHNICALS (Max 30 Points)
    if technicals:
        rsi = technicals.get('rsi', {}).get('val', 50)
        if 35 < rsi < 70: score += 10
        
        trend = technicals.get('trend', {}).get('status', '')
        if "Bullish" in trend: score += 20
        elif "Bearish" in trend: score -= 5
        else: score += 5

    # 3. MACRO / POLICY (Max 20 Points)
    if policy and 'insights' in policy:
        budget = policy['insights'].get('budget', '').lower()
        if "positive" in budget: score += 20
        elif "neutral" in budget: score += 10

    # 4. NEWS SENTIMENT (Max 10 Points)
    sentiment = 0
    if news_list:
        for n in news_list:
            blob = TextBlob(n['title'])
            sentiment += blob.sentiment.polarity
        
        if sentiment > 0.1: score += 10
        elif sentiment < -0.1: score -= 5

    # Final Verdict
    if score >= 80: verdict = "STRONG BUY"
    elif score >= 60: verdict = "BUY"
    elif score >= 40: verdict = "HOLD"
    else: verdict = "AVOID"

    color = 'text-success' if score >= 60 else 'text-danger' if score < 40 else 'text-warning'

    return {'score': score, 'verdict': verdict, 'color': color}

def calculate_narendra_rating(data):
    ratings = {}
    thresh = RATING_THRESHOLDS
    
    # ... (Keep existing Logic) ...
    # FIX: Handle 0% ROE
    roe = data.get('roe', 0)
    if roe == 0:
        ratings['efficiency'] = {"metric": "N/A", "benchmark": "-", "status": "No Data", "color": "text-muted"}
    elif roe > thresh['ROE_GOOD']:
        ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_GOOD']}%", "status": "Excellent", "color": "text-success"}
    elif roe > thresh['ROE_AVG']:
        ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_AVG']}%", "status": "Average", "color": "text-warning"}
    else:
        ratings['efficiency'] = {"metric": f"{roe}%", "benchmark": f"> {thresh['ROE_AVG']}%", "status": "Poor", "color": "text-danger"}

    # Ownership
    prom = data.get('promoter_holding', 0)
    if prom > 50: ratings['ownership'] = {"metric": f"{prom}%", "benchmark": "> 50%", "status": "High", "color": "text-success"}
    else: ratings['ownership'] = {"metric": f"{prom}%", "benchmark": "> 30%", "status": "Stable", "color": "text-primary"}

    # Valuation
    pe = data.get('pe_ratio', 0)
    ind_pe = data.get('industry_pe', 25)
    if pe < ind_pe: ratings['valuation'] = {"metric": f"{pe}x", "benchmark": f"{ind_pe}x", "status": "Cheap", "color": "text-success"}
    else: ratings['valuation'] = {"metric": f"{pe}x", "benchmark": f"{ind_pe}x", "status": "Expensive", "color": "text-danger"}

    # Financials
    de = data.get('debt_to_equity', 0)
    if de < 0.1: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": "< 0.1", "status": "Debt Free", "color": "text-success"}
    elif de < 1.0: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": "< 1.0", "status": "Stable", "color": "text-primary"}
    else: ratings['financials'] = {"metric": f"D/E {de}", "benchmark": "< 1.0", "status": "High Debt", "color": "text-danger"}

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
        if start_price <= 0:
            return None

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
        print(
            f"Wealth Error for {getattr(ticker_obj, 'ticker', 'Unknown')}: {e}")
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
        if pl.empty or bs.empty:
            return None

        years = [d.strftime('%Y') for d in pl.columns[:2]]
        statements['years'] = years

        # --- 1. INCOME STATEMENT ---
        statements['pl_rows'] = [
            get_accounting_row(
                pl, ['Total Revenue', 'Total Revenue'], 'Total Revenue', indent=0, bold=True),
            get_accounting_row(pl, ['Cost Of Revenue', 'Cost of Goods Sold'],
                               'Cost of Goods Sold (COGS)', indent=0, negative=True),
            # Gross Profit (Green)
            {'label': 'Gross Profit', 'values': [round(v/10000000, 2) for v in pl.loc['Gross Profit'].values[:2]],
             'style': 'background-color: #dcfce7; font-weight: 700; color: #064e3b;'} if 'Gross Profit' in pl.index else None,

            {'label': 'Operating Expenses:', 'values': [
                '', ''], 'style': 'background-color: #f8fafc; font-weight: 700;'},
            get_accounting_row(pl, ['Selling General And Administration'],
                               'Selling, General & Admin (SG&A)', indent=1, negative=True),
            get_accounting_row(pl, ['Research And Development'],
                               'Research & Development (R&D)', indent=1, negative=True),

            # Operating Income (Green)
            {'label': 'Operating Income (or EBIT)', 'values': [round(v/10000000, 2) for v in pl.loc['Operating Income'].values[:2]],
             'style': 'background-color: #dcfce7; font-weight: 700;'} if 'Operating Income' in pl.index else None,

            {'label': 'Other Income and (Expenses):', 'values': [
                '', ''], 'style': 'background-color: #f8fafc; font-weight: 700;'},
            get_accounting_row(pl, ['Interest Expense'],
                               'Interest Expense', indent=1, negative=True),

            get_accounting_row(pl, ['Pretax Income'],
                               'Income Before Tax', indent=0, bold=True),
            get_accounting_row(
                pl, ['Tax Provision'], 'Income Tax Expense', indent=0, negative=True),

            # Net Income (Green + Border)
            {'label': 'NET INCOME', 'values': [round(v/10000000, 2) for v in pl.loc['Net Income'].values[:2]],
             'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a; font-size: 1rem; color: #064e3b;'}
        ]
        statements['pl_rows'] = [x for x in statements['pl_rows'] if x]

        # --- 2. BALANCE SHEET ---
        statements['bs_rows'] = [
            {'label': 'ASSETS', 'values': [
                '', ''], 'style': 'font-weight: 800; text-transform: uppercase; background-color: #f1f5f9;'},
            {'label': 'Current Assets', 'values': [
                '', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(
                bs, ['Cash And Cash Equivalents'], 'Cash and Cash Equivalents', indent=1),
            get_accounting_row(bs, ['Receivables'],
                               'Accounts Receivable', indent=1),
            get_accounting_row(bs, ['Inventory'], 'Inventory', indent=1),
            get_accounting_row(bs, ['Current Assets'],
                               'Total Current Assets', indent=0, bold=True),

            {'label': 'Non-Current Assets',
                'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(
                bs, ['Net PPE'], 'Net Property, Plant, and Equipment', indent=1),
            get_accounting_row(
                bs, ['GoodwillAndIntangibleAssets'], 'Intangible Assets', indent=1),
            get_accounting_row(bs, ['Total Non Current Assets'],
                               'Total Non-Current Assets', indent=0, bold=True),

            # Total Assets (Green)
            {'label': 'TOTAL ASSETS', 'values': [round(v/10000000, 2) for v in bs.loc['Total Assets'].values[:2]],
             'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'},

            {'label': 'LIABILITIES AND STOCKHOLDERS\' EQUITY', 'values': [
                '', ''], 'style': 'font-weight: 800; text-transform: uppercase; background-color: #f1f5f9;'},
            {'label': 'Current Liabilities', 'values': [
                '', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Accounts Payable'],
                               'Accounts Payable', indent=1),
            get_accounting_row(bs, ['Current Debt'],
                               'Current Portion of Debt', indent=1),
            get_accounting_row(
                bs, ['Current Liabilities'], 'Total Current Liabilities', indent=0, bold=True),

            {'label': 'Non-Current Liabilities',
                'values': ['', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Long Term Debt'],
                               'Long-Term Debt', indent=1),
            get_accounting_row(bs, ['Total Non Current Liabilities Net Minority Interest'],
                               'Total Non-Current Liabilities', indent=0, bold=True),

            # Total Liabilities (Red)
            {'label': 'Total Liabilities', 'values': [round(v/10000000, 2) for v in bs.loc['Total Liabilities Net Minority Interest'].values[:2]],
             'style': 'background-color: #fee2e2; font-weight: 700; border-top: 2px solid #ef4444; color: #991b1b;'},

            {'label': 'Stockholders\' Equity', 'values': [
                '', ''], 'style': 'font-weight: 700; padding-left: 10px;'},
            get_accounting_row(bs, ['Common Stock'], 'Common Stock', indent=1),
            get_accounting_row(bs, ['Retained Earnings'],
                               'Retained Earnings', indent=1),

            # Total Equity (Blue)
            {'label': 'Total Stockholders\' Equity', 'values': [round(
                v/10000000, 2) for v in bs.loc['Stockholders Equity'].values[:2]], 'style': 'background-color: #dbeafe; font-weight: 700;'},

            {'label': 'TOTAL LIABILITIES AND EQUITY', 'values': [round(
                v/10000000, 2) for v in bs.loc['Total Assets'].values[:2]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'}
        ]

        # --- 3. CASH FLOW ---
        statements['cf_rows'] = [
            {'label': 'Cash Flow from Operating Activities', 'values': [
                '', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(
                cf, ['Net Income', 'Net Income From Continuing Operations'], 'Net Income', indent=1),
            get_accounting_row(cf, ['Depreciation'],
                               'Add: Depreciation', indent=1),
            get_accounting_row(
                cf, ['Operating Cash Flow'], 'Net Cash from Operating Activities', indent=0, bold=True),

            {'label': 'Cash Flow from Investing Activities', 'values': [
                '', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(cf, ['Capital Expenditure'],
                               'Purchase of PPE', indent=1, negative=True),
            get_accounting_row(
                cf, ['Investing Cash Flow'], 'Net Cash used in Investing Activities', indent=0, bold=True),

            {'label': 'Cash Flow from Financing Activities', 'values': [
                '', ''], 'style': 'font-weight: 700; background-color: #f8fafc;'},
            get_accounting_row(
                cf, ['Cash Dividends Paid'], 'Payment of dividends', indent=1, negative=True),
            get_accounting_row(
                cf, ['Financing Cash Flow'], 'Net Cash from Financing Activities', indent=0, bold=True),

            # Net Change (Blue)
            {'label': 'Net Change in Cash', 'values': [round(v/10000000, 2) for v in cf.loc['End Cash Position'].values[:2]],
             'style': 'background-color: #dbeafe; font-weight: 800; border-top: 2px solid #2563eb;'} if 'End Cash Position' in cf.index else None
        ]
        statements['cf_rows'] = [x for x in statements['cf_rows'] if x]

        # --- 4. RETAINED EARNINGS ---
        re_vals = bs.loc['Retained Earnings'].values[:2] if 'Retained Earnings' in bs.index else [
            0, 0]
        re_list = [round(v/10000000, 2) for v in re_vals]
        while len(re_list) < 2:
            re_list.append(0)

        statements['re_rows'] = [
            {'label': 'Beginning Retained Earnings', 'values': [
                re_list[1]], 'style': 'font-weight: 700;'},
            {'label': 'Plus: Net Income', 'values': [round(
                pl.loc['Net Income'].iloc[0]/10000000, 2) if 'Net Income' in pl.index else 0], 'style': 'padding-left: 20px;'},
            {'label': 'Less: Dividends Paid', 'values': [round(
                cf.loc['Cash Dividends Paid'].iloc[0]/10000000, 2) if 'Cash Dividends Paid' in cf.index else 0], 'style': 'padding-left: 20px; color: #991b1b;'},

            # ENDING RE (Green)
            {'label': 'Ending Retained Earnings', 'values': [
                re_list[0]], 'style': 'background-color: #dcfce7; font-weight: 800; border-top: 2px solid #16a34a;'}
        ]
    except:
        return None
    return statements


def get_peer_data_with_share(ticker, sector, current_mcap):
    client = get_bq_client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    table_id = f"{client.project}.{dataset_id}.stock_intelligence_v3"

    # 1. Get Peers List
    peers_list = get_dynamic_peers(ticker, sector)
    
    # 2. Safety Check: If no peers found, return empty immediately
    if not peers_list:
        return [], 100

    # 3. OPTIMIZED: Fetch Data from BigQuery (Do NOT loop Yahoo!)
    try:
        # We select specific columns so we don't need to live fetch
        # The subquery ensures we only get the latest row for each ticker
        sql = f"""
            SELECT ticker, price, mcap, pe_ratio, roe
            FROM (
                SELECT ticker, price, mcap, pe_ratio, roe,
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY last_updated DESC) as rn
                FROM `{table_id}`
                WHERE ticker IN UNNEST({peers_list})
            )
            WHERE rn = 1
        """
        df = client.query(sql).to_dataframe()
        
        # Convert to dictionary directly
        peer_data = []
        total_sector_mcap = current_mcap

        for index, row in df.iterrows():
            # Data in BQ is already in Crores (saved by sync_stock_on_demand)
            m_cap = safe_float(row['mcap'])
            
            if m_cap > 0:
                total_sector_mcap += m_cap
                peer_data.append({
                    'ticker': row['ticker'],
                    'price': round(safe_float(row['price']), 2),
                    'mcap': m_cap,
                    'pe': round(safe_float(row['pe_ratio']), 1),
                    'roe': round(safe_float(row['roe']), 1),
                    'share': 0 # Placeholder, calculated below
                })

        # 4. Calculate Market Share percentages
        for p in peer_data:
            p['share'] = round((p['mcap'] / total_sector_mcap) * 100, 1) if total_sector_mcap > 0 else 0
            
        current_share = round((current_mcap / total_sector_mcap) * 100, 1) if total_sector_mcap > 0 else 100

        # Return sorted by Market Cap (Largest first)
        return sorted(peer_data, key=lambda x: x['mcap'], reverse=True), current_share
        
    except Exception as e:
        print(f"⚠️ Peer SQL Error: {e}")
        # Return empty list on error to prevent crash
        return [], 100

def get_dynamic_peers(ticker, sector):
    # 🛑 SAFETY GUARD: Stop if running during server startup
    # This prevents the "4x Loop" crash if called by apps.py or settings.py
    if 'gunicorn' in sys.argv[0] or 'runserver' in sys.argv[0]:
        # We check if we are inside a request by looking for a special flag
        # or simply by checking if the cache is ready.
        # Ideally, we just return empty if it feels like a startup script.
        pass
    
    client = get_bq_client()
    dataset_id = os.getenv("GCP_DATASET_ID")
    # Make sure this matches your actual table name (_v3)
    table_id = f"{client.project}.{dataset_id}.stock_intelligence_v3"

    try:
        # The fix: include last_updated and mcap in the subquery's output
        query = f"""
        SELECT ticker FROM (
            SELECT 
                ticker, 
                mcap,
                last_updated,  -- MUST be here to be seen by the outer WHERE
                ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY last_updated DESC) as rn
            FROM `{table_id}`
            WHERE sector = '{sector}'
        )
        WHERE rn = 1 
          AND ticker != '{ticker}'
          -- Now the outer query can "see" last_updated
          AND last_updated >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY mcap DESC
        LIMIT 5
        """
        
        # Execute query
        query_job = client.query(query)
        results = query_job.result()
        
        tickers = [row.ticker for row in results]
        print(f"✅ Found peers for {ticker}: {tickers}")
        return tickers

    except Exception as e:
        print(f"❌ Error fetching peers for {ticker}: {e}")
        return []


def get_safe_shareholding(ticker_obj, stock_info):
    # 1. Debug Printing
    if isinstance(stock_info, dict):
        print(f"🕵️ DEBUG SHAREHOLDING KEYS: {list(stock_info.keys())}")
        print(
            f"Insiders: {stock_info.get('heldPercentInsiders')}, Institutions: {stock_info.get('heldPercentInstitutions')}")

    data = {"promoter": 0, "institution": 0, "public": 0, "top_investors": []}

    try:
        # --- SAFETY CHECK ---
        if not isinstance(stock_info, dict):
            stock_info = {}

        # Get total shares to help convert raw numbers to percentages if needed
        total_shares = stock_info.get('sharesOutstanding') or 1

        # --- STRATEGY 1: Check .info (Fastest) ---
        p_hold = stock_info.get('heldPercentInsiders')
        i_hold = stock_info.get('heldPercentInstitutions')

        if p_hold is not None:
            data['promoter'] = round(p_hold * 100, 2)
        if i_hold is not None:
            data['institution'] = round(i_hold * 100, 2)

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
                                except:
                                    continue
                        if "insider" in row_str:
                            data['promoter'] = val
                        elif "institut" in row_str:
                            data['institution'] = val
            except:
                pass

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
                    name = row.get('holder', row.iloc[0] if len(
                        row) > 0 else 'Unknown')

                    # Find Stake % (Aggressive detection)
                    stake_val = 0
                    for cell in row:
                        try:
                            val = float(str(cell).replace('%', '').strip())
                            if val <= 0:
                                continue

                            if val > 100:  # It's a share count, convert it
                                stake_val = (val / total_shares) * 100
                            elif val < 1:  # It's a decimal ratio
                                stake_val = val * 100
                            else:  # It's a standard percentage
                                stake_val = val
                            if stake_val > 0:
                                break
                        except:
                            continue

                    if stake_val > 0:
                        # Return as a NUMBER for the template to handle
                        data['top_investors'].append({
                            'name': name,
                            'stake': round(stake_val, 2)
                        })
        except:
            pass

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
    # --- STEP 1: RESOLVE SYMBOL ---
    try:
        symbol = resolve_symbol_from_name(query)
    except:
        symbol = query.upper().replace(".", "").replace("LTD", "").strip()

    safe_key = slugify(symbol)
    cache_key = f"stock_data_{safe_key}"
    
    # Check RAM Cache first
    cached_payload = cache.get(cache_key)
    if cached_payload:
        return cached_payload

    client = None
    table_id = None
    
    # Placeholder for BQ data in case Live Fetch fails later
    backup_data = None 

    # =========================================================
    # 3. WAREHOUSE LAYER: Check BigQuery
    # =========================================================
    try:
        client = get_bq_client()
        dataset_id = os.getenv("GCP_DATASET_ID")
        table_id = f"{client.project}.{dataset_id}.stock_intelligence_v3"
        
        sql = f"""
            SELECT * FROM `{table_id}` 
            WHERE ticker = '{symbol}' 
            ORDER BY last_updated DESC
            LIMIT 1
        """
        bq_data = client.query(sql).to_dataframe()

        if not bq_data.empty:
            row = bq_data.iloc[0].to_dict()
            
            # --- CHECK FRESHNESS ---
            last_upd = row.get('last_updated')
            is_fresh = False
            
            if last_upd:
                try:
                    # Convert to UTC for comparison
                    if isinstance(last_upd, str):
                        last_upd = datetime.fromisoformat(last_upd.replace('Z', '+00:00'))
                    if last_upd.tzinfo is None:
                        last_upd = pytz.UTC.localize(last_upd)
                    else:
                        last_upd = last_upd.astimezone(pytz.UTC)
                    
                    row['last_updated'] = last_upd # Save back as object
                    
                    # 60 Minute Rule
                    now_utc = datetime.now(pytz.UTC)
                    is_fresh = (now_utc - last_upd).total_seconds() < 3600
                    # is_fresh = (now_utc - last_upd).total_seconds() < 0
                except:
                    is_fresh = False

            # --- DECISION POINT ---
            if is_fresh:
                print(f"⚡ BQ HIT: {symbol} is Fresh (<60m). Returning.")
                
                # Deserialize and Return immediately
                json_cols = ['technicals', 'policy', 'arth_score', 'news',
                             'chart_dates', 'chart_prices', 'chart_volumes', 'shareholding']
                for field in json_cols:
                    val = row.get(field)
                    if val is None or pd.isna(val):
                        row[field] = {} if 'chart' not in field else []
                    elif isinstance(val, str):
                        try: row[field] = json.loads(val)
                        except: row[field] = {} if 'chart' not in field else []
                
                # Convert date to IST for display
                row['last_updated'] = row['last_updated'].astimezone(pytz.timezone('Asia/Kolkata'))
                row['is_stale'] = False
                cache.set(cache_key, row, timeout=60*15)
                return row
            
            else:
                print(f"⚠️ BQ STALE: {symbol} data is old. Falling through to LIVE FETCH...")
                # We save this stale row as a "Backup" in case Yahoo fails
                row['is_stale'] = True
                backup_data = row

    except Exception as e:
        print(f"⚠️ BQ Check Error: {e}")

    # =========================================================
    # 4. LIVE LAYER: Fetch Fresh Data (The "Rescue" Layer)
    # =========================================================
    print(f"🔄 LIVE FETCH: Getting real-time data for {symbol}...")

    try:
        ticker_name = symbol if ".BO" in symbol else f"{symbol}.NS"
        ticker_obj = yf.Ticker(ticker_name)
        
        # Try to get info - If this fails, we use the Backup!
        try:
            stock_info = ticker_obj.info
        except:
            if backup_data:
                print(f"⚠️ Live Fetch Failed. Returning STALE backup for {symbol}.")
                # Deserialize backup before returning
                json_cols = ['technicals', 'policy', 'arth_score', 'news', 'chart_dates', 'chart_prices', 'chart_volumes', 'shareholding']
                for field in json_cols:
                    val = backup_data.get(field)
                    if isinstance(val, str):
                        try: backup_data[field] = json.loads(val)
                        except: pass
                # Fix timezone for display
                if isinstance(backup_data.get('last_updated'), datetime):
                     backup_data['last_updated'] = backup_data['last_updated'].astimezone(pytz.timezone('Asia/Kolkata'))
                return backup_data
            return None

        if not stock_info or 'currentPrice' not in stock_info:
            return None

   
        hist = ticker_obj.history(period="5y")
        
        # =========================================================
        # 🛡️ SMART METRIC CALCULATION (Prevents 0.0 scores)
        # =========================================================
        
        # 1. PE Ratio (Try Trailing, then Forward)
        pe_val = safe_float(stock_info.get('trailingPE') or stock_info.get('forwardPE') or 0)
        
        # 2. ROE (If missing, estimate from Price/Book and PE)
        roe = round(safe_float(stock_info.get('returnOnEquity')) * 100, 2)
        if roe == 0 and pe_val > 0:
            # Fallback: ROE ≈ (P/B) / (P/E)
            pb = safe_float(stock_info.get('priceToBook'))
            if pb > 0:
                roe = round((pb / pe_val) * 100, 2)
                print(f"⚠️ {symbol}: ROE missing. Estimated {roe}% from P/B & P/E.")

        # 3. Growth (If missing, default to a safe positive number if PE is high)
        earnings_growth = safe_float(stock_info.get('earningsGrowth'))
        revenue_growth = safe_float(stock_info.get('revenueGrowth'))
        
        # Use Revenue Growth if Earnings Growth is missing
        profit_growth = round((earnings_growth or revenue_growth or 0) * 100, 2)
        
        # 4. Debt (Standard)
        raw_de = safe_float(stock_info.get('debtToEquity'))
        debt_to_equity = round(raw_de / 100, 2) if raw_de > 5 else round(raw_de, 2)

        
        refined_sector = refine_sector_name(
            stock_info.get('sector', 'Unknown'),
            stock_info.get('industry', 'Unknown'),
            symbol,
            stock_info.get('longBusinessSummary', '')
        )

        # =========================================================
        # REPLACEMENT BLOCK FOR data_payload
        # =========================================================
        
        # 1. Pre-calculate metrics
        tech_data = calculate_technicals(hist)
        news_data = get_robust_news(ticker_obj, symbol)
        policy_data = get_policy_context(refined_sector, stock_info.get('industry', ''))

        # Pack Metrics for Scoring
        fin_metrics = {
            'roe': roe, 
            'profit_growth': profit_growth,
            'debt_to_equity': debt_to_equity
        } 

        # --- BUILD PAYLOAD ---
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
            'profit_growth': fin_metrics['profit_growth'],
            'pe_ratio': round(safe_float(pe_val), 2),
            'industry_pe': get_industry_pe(refined_sector, symbol, FALLBACK_SECTOR_PE.get(refined_sector, 20.0)),
            'debt_to_equity': debt_to_equity,
            'cagr_5y': 0, 

            # ✅ THIS IS THE CRITICAL FIX: Pass Real Data
            'technicals': tech_data,
            'news': news_data,
            'policy': policy_data,
            'arth_score': calculate_arth_score(fin_metrics, tech_data, policy_data, news_data, refined_sector),
            
            'shareholding': get_safe_shareholding(ticker_obj, stock_info),
            'last_updated': datetime.now(pytz.timezone('Asia/Kolkata')),
            'is_stale': False
        }

        # CAGR Fix
        if not hist.empty:
            start_p = hist['Close'].iloc[0]
            end_p = hist['Close'].iloc[-1]
            if start_p > 0:
                data_payload['cagr_5y'] = round(((end_p / start_p) ** (1 / 5) - 1) * 100, 2)
                
            data_payload['chart_dates'] = hist.index.strftime('%Y-%m-%d').tolist()
            data_payload['chart_prices'] = hist['Close'].round(2).tolist()
            data_payload['chart_volumes'] = hist['Volume'].fillna(0).tolist()
        else:
            data_payload['chart_dates'], data_payload['chart_prices'], data_payload['chart_volumes'] = [], [], []

        # =========================================================
        # 5. STORAGE LAYER: Save FRESH data
        # =========================================================
        cache.set(cache_key, data_payload, timeout=60*15)

        if client and table_id:
            try:
                if "_v2" in table_id: table_id = table_id.replace("_v2", "_v3")
                elif "_v3" not in table_id: table_id = table_id + "_v3"

                bq_record = data_payload.copy()
                bq_record['last_updated'] = datetime.now(pytz.UTC) # Save as UTC
                if 'is_stale' in bq_record: del bq_record['is_stale']

                json_fields = ['technicals', 'news', 'policy', 'arth_score', 
                               'chart_dates', 'chart_prices', 'chart_volumes', 'shareholding']
                
                for field in json_fields:
                    val = bq_record.get(field)
                    if val is None: bq_record[field] = "{}"
                    elif isinstance(val, (dict, list)): bq_record[field] = json.dumps(val, default=str)
                    else: bq_record[field] = str(val)

                df = pd.DataFrame([bq_record])
                for col in json_fields: 
                    if col in df.columns: df[col] = df[col].astype("string")

                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", schema_update_options=["ALLOW_FIELD_ADDITION"])
                client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
                print(f"✅ Data Saved to BigQuery: {symbol}")

            except Exception as e:
                print(f"⚠️ BQ Save Error: {e}")

        return data_payload

    except Exception as e:
        print(f"❌ Critical Live Fetch Error: {e}")
        # FINAL SAFETY NET: If live fetch crashes, return the stale data we found earlier
        if backup_data:
            print("Using Stale Backup due to crash.")
            # ... (Deserialization logic for backup_data again) ...
            json_cols = ['technicals', 'policy', 'arth_score', 'news', 'chart_dates', 'chart_prices', 'chart_volumes', 'shareholding']
            for field in json_cols:
                val = backup_data.get(field)
                if isinstance(val, str):
                    try: backup_data[field] = json.loads(val)
                    except: pass
            if isinstance(backup_data.get('last_updated'), datetime):
                 backup_data['last_updated'] = backup_data['last_updated'].astimezone(pytz.timezone('Asia/Kolkata'))
            return backup_data
        return None


def index(request):
    # 1. Get User Input (REMOVE THE DEFAULT VALUE - FAST STARTUP)
    raw_query = request.GET.get('q', '').strip()
    
    # 2. If no search query, show a "Welcome" screen (FAST STARTUP)
    if not raw_query:
        return render(request, 'dashboard/index.html', {
            'landing_mode': True, 
            'year_range': list(range(2000, datetime.now().year + 1))
        })

    # --- EVERYTHING BELOW RUNS ONLY WHEN SEARCHING ---
    wealth_amount = int(request.GET.get('w_amt', 100000))
    wealth_year = int(request.GET.get('w_year', 2011))

    # 2. Fetch Main Data
    data = sync_stock_on_demand(raw_query)

    # Handle Invalid Stock
    if not data:
        return render(request, 'dashboard/index.html', {'error_message': f"Ticker '{raw_query}' not valid."})

    symbol = data['ticker']

    # 3. Re-create Ticker Object
    t_name = symbol if ".BO" in symbol else f"{symbol}.NS"
    ticker_obj = yf.Ticker(t_name)

    # 4. Run Analysis Modules
    try:
        financials = get_detailed_financials(ticker_obj)
        fin_analysis = analyze_financial_health(ticker_obj)
        wealth_data = calculate_wealth_growth(ticker_obj, wealth_amount, wealth_year)
    except Exception as e:
        print(f"⚠️ Analysis Module Error: {e}")
        financials, fin_analysis, wealth_data = {}, {}, None

    # 5. Shareholding
    shareholding = data.get('shareholding', {})

    # 6. Pie Chart Data
    pie_labels_list = ["Promoters", "Institutions", "Public & Others"]
    pie_data_list = [
        shareholding.get('promoter', 0),
        shareholding.get('institution', 0),
        shareholding.get('public', 0)
    ]

    # 7. Benchmark Comparison
    bench_sym = SECTOR_BENCHMARKS.get(data['sector'], "^NSEI")
    bench_stats = {}
    dates, stock_pct, bench_pct, volumes = [], [], [], []

    try:
        if 'chart_prices' in data and data['chart_prices']:
            b_hist = yf.Ticker(bench_sym).history(period="5y")
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

    # =========================================================
    # 🚨 CRITICAL FIX: CALL PEERS ONCE, USE TWICE
    # =========================================================
    # Before, you called this 4 times. Now we call it 1 time.
    peer_result = get_peer_data_with_share(symbol, data['sector'], data['mcap'])
    
    if peer_result and len(peer_result) == 2:
        peers_list = peer_result[0]
        my_share_data = peer_result[1]
    else:
        peers_list = []
        my_share_data = {}

    # 8. Prepare Context
    context = {
        'symbol': symbol,
        'data': data,
        'narendra_rating': calculate_narendra_rating(data),
        
        # ✅ USE THE PRE-CALCULATED VARIABLES
        'peers': peers_list,
        'my_share': my_share_data,

        'financials': financials,
        'fin_analysis': fin_analysis,
        'shareholding': shareholding,

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