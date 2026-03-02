# Arth-Insight: Financial Analytics Engine

[![Deployment Status](https://img.shields.io/badge/Deployment-Live-success?style=flat-square&logo=render)](https://arth-insight-ding.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![dbt](https://img.shields.io/badge/Transform-dbt-FF694B?style=flat-square&logo=dbt)](https://www.getdbt.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

**Arth-Insight** is a full-stack equity research platform engineered to democratize institutional-grade financial analysis. It leverages a **Hybrid Data Architecture** to deliver real-time NSE market insights, proprietary scoring models, and macro-economic context with sub-second latency.

🚀 **Live Terminal:** [https://arth-insight.onrender.com](https://arth-insight-ding.onrender.com)

---

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technical Highlights](#-technical-highlights)
- [Modern Data Stack (dbt & BigQuery)](#-modern-data-stack-dbt--bigquery)
- [Automated Pipelines](#-automated-pipelines)
- [Installation & Docker Support](#-installation--docker-support)
- [Limitations & Constraints](#-limitations--constraints)
- [Future Roadmap](#-future-roadmap)

---

## 🏗 System Architecture

The application addresses the classic trade-off between **Data Freshness** and **System Latency** using a dual-layer data strategy:

1.  **Hot Path (Live Layer):**
    * Fetches real-time price, volume, and technical indicators via the Yahoo Finance API.
    * Used for volatile data points that require second-level precision (e.g., CMP, Day Change).
2.  **Cold Path (Warehouse Layer):**
    * Stores fundamental data (P/E, ROE, Market Cap) and sector classifications in **Google BigQuery**.
    * Acts as a high-performance cache. When a user requests peer comparisons, the system queries the warehouse (taking ~30ms) instead of making multiple external API calls (taking 40s+).
3.  **Smart Caching Protocol (Market-Aware):**
    * **Dynamic Holiday Check:** Uses `pandas_market_calendars` to automatically detect NSE holidays and weekends.
    * **Logic:** If the market is CLOSED, the system skips the live API fetch and serves instant "Last Closing Price" data from BigQuery.
    * **Result:** Zero latency on weekends and holidays.

---

## 🌟 Key Features

### 1. The "Arth-Verdict" Scoring Engine (V3)
A proprietary algorithm powered by **dbt** that condenses complex financial metrics into a single **0-100 Health Score**.
* **Sector-Aware Logic:** Automatically adapts scoring rules. For example, it penalizes high debt for Manufacturing companies but ignores it for Banks/NBFCs (where leverage is operational).
* **Recovery Detection:** Identifies turnaround candidates by rewarding companies with high ROE even if immediate growth data is missing.

### 2. Dynamic Peer Intelligence
* **Algorithm:** Real-time identification of sector competitors based on market cap and business classification.
* **Metric:** Calculates "Relative Valuation" and "Market Share" dynamically without manual tagging.

### 3. Macro-Economic Context Layer
* Maps individual stocks to broader economic indicators (RBI Repo Rate, Union Budget allocations, PLI Schemes).
* *Example:* A search for "Suzlon" automatically pulls relevant "Power & Renewable Energy" policy updates.

### 4. Wealth Growth Simulator
* An interactive financial model visualizing the compounding effect of lumpsum investments over historical timeframes (5Y, 10Y, Max).
* **Cached Performance:** Wealth calculations are cached for 24 hours to prevent redundant computations on weekends.

---

## 🔧 Technical Highlights

* **Optimized SQL Queries:** Replaced iterative N+1 API calls for peer data with a single, vectorized BigQuery `SELECT ... WHERE ticker IN (...)` operation, reducing page load time by **92%**.
* **Robust Sector Mapping:** Implemented a "God Mode" classifier that overrides generic API labels (e.g., correcting "Shriram Finance" from *Infrastructure* to *NBFC*) to ensure accurate peer comparison.
* **Fault Tolerance:** Includes hardcoded fallbacks for critical policy data and "Smart Estimation" for missing financial ratios.

---

## 📊 Modern Data Stack (dbt & BigQuery)

Arth-Insight has migrated to a professional **ELT (Extract, Load, Transform)** architecture:

### 1. The "Raw" Layer (`stock_raw_data`)
* **Ingestion:** GitHub Actions fetches raw data from Yahoo Finance hourly.
* **Destination:** Data lands in the `stock_intelligence_v3` table in BigQuery.

### 2. The "Transformation" Layer (dbt)
* **Tool:** **dbt (Data Build Tool)** manages all business logic.
* **Models:**
    * `stg_stocks.sql`: Cleans raw data, handles nulls, and standardizes column names.
    * `fact_stock_analysis.sql`: Applies the V3 Scoring Logic (Sector-specific rules) and generates the final "Gold" table.
* **Orchestration:** dbt Cloud runs daily jobs to refresh the scoring models.

### 3. The "Production" Layer (`prod`)
* **Serving:** The Django application connects strictly to the `prod` dataset.
* **Safety:** "Source Freshness" tests ensure that stale data never reaches the production dashboard.

> 🔗 **Data Transformations:** The SQL logic for this project is managed in a separate repository: [https://github.com/nbx0021/arth-insight-dbt.git].

---

## 🚀 Automated Pipelines

The project implements a full **DevOps & DataOps Pipeline**:

### 1. CI/CD (GitHub Actions)
* **Workflow:** `.github/workflows/docker_push.yml`
* **Trigger:** Activates automatically on every `git push` to the `main` branch.
* **Action:** Builds the Docker image and pushes it to Docker Hub, triggering a live redeploy on Render.

### 2. Market Data ETL
* **Workflow:** `.github/workflows/docker_market_etl.yml`
* **Schedule:** Runs strictly during **NSE Market Hours** (9:15 AM - 3:30 PM IST).
* **Logic:**
    1.  Validates market status (Open/Closed/Holiday).
    2.  Fetches the Nifty 500 universe.
    3.  Performs batch processing to update fundamental metrics in BigQuery.

---

## 📦 Installation & Docker Support

This project is fully containerized for consistent deployment across environments.

### Prerequisites
* Docker & Docker Compose
* Google Cloud Platform (GCP) Service Account Credentials

### Running with Docker (Recommended)

```bash
# 1. Clone the repository
git clone [https://github.com/nbx0021/arth-insight.git](https://github.com/nbx0021/arth-insight.git)
cd arth-insight

# 2. Build the Docker image
docker build -t arth-insight .

# 3. Run the container
# Ensure your .env file contains valid GCP credentials
docker run -p 8000:8000 --env-file .env arth-insight

```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
python src/portal/manage.py migrate

# Start server
python src/portal/manage.py runserver

```

---

## ⚠️ Limitations & Constraints

Transparency regarding the current infrastructure (Free Tier):

1. **Cold Start Latency:** Hosted on **Render Free Tier**. The application "sleeps" after inactivity, causing a 30-50 second delay on the very first request.
2. **API Rate Limits:** Relies on public financial APIs. High-frequency automated scraping may lead to temporary IP throttling.
3. **Data Granularity:** Intraday charts are limited to 1-minute intervals.

---

## 🔮 Future Roadmap

* **v2.0:** Integrate Large Language Models (LLMs) to summarize quarterly earnings call transcripts.
* **v2.1:** User Authentication system allowing persistent portfolios and price alert configurations.
* **v2.2:** Implementation of a Technical Screener (e.g., "Show all stocks with RSI < 30").

---

**Author:** Narendra Bhandari
*Full Stack Developer & Data Engineer*
