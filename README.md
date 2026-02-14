
# Arth-Insight: Financial Analytics Engine

[![Deployment Status](https://img.shields.io/badge/Deployment-Live-success?style=flat-square&logo=render)](https://arth-insight-ding.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

**Arth-Insight** is a full-stack equity research platform engineered to democratize institutional-grade financial analysis. It leverages a **Hybrid Data Architecture** to deliver real-time NSE market insights, proprietary scoring models, and macro-economic context with sub-second latency.

🚀 **Live Terminal:** [https://arth-insight.onrender.com](https://arth-insight-ding.onrender.com)

---

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technical Highlights](#-technical-highlights)
- [Automated ETL Pipelines](#-automated-etl-pipelines)
- [Installation & Docker Support](#-installation--docker-support)
- [Limitations & Constraints](#-limitations--constraints)
- [Future Roadmap](#-future-roadmap)

---

## 🏗 System Architecture

The application addresses the classic trade-off between **Data Freshness** and **System Latency** using a dual-layer data strategy:

1.  **Hot Path (Live Layer):**
    * Fetches real-time price, volume, and technical indicators via the Yahoo Finance API.
    * Used for volatile data points that require second-level precision.
2.  **Cold Path (Warehouse Layer):**
    * Stores fundamental data (P/E, ROE, Market Cap) and sector classifications in **Google BigQuery**.
    * Acts as a high-performance cache. When a user requests peer comparisons, the system queries the warehouse (taking ~30ms) instead of making multiple external API calls (taking 40s+).
3.  **Smart Caching Protocol:**
    * Implements a "Staleness Check": Data < 60 minutes old is served from the database/cache. Data > 60 minutes old triggers a background refresh.

---

## 🌟 Key Features

### 1. The "Arth-Verdict" Scoring Engine
A proprietary algorithm that condenses complex financial metrics into a single **0-100 Health Score**.
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

---

## 🔧 Technical Highlights

* **Optimized SQL Queries:** Replaced iterative N+1 API calls for peer data with a single, vectorized BigQuery `SELECT ... WHERE ticker IN (...)` operation, reducing page load time by **92%**.
* **Robust Sector Mapping:** Implemented a "God Mode" classifier that overrides generic API labels (e.g., correcting "Shriram Finance" from *Infrastructure* to *NBFC*) to ensure accurate peer comparison.
* **Fault Tolerance:** Includes hardcoded fallbacks for critical policy data and "Smart Estimation" for missing financial ratios (e.g., deriving ROE from P/B and P/E when direct data is unavailable).

--- 
### 🚀 Automated Deployment (CI/CD) and 🔄 Automated ETL Pipelines

The project implements a full **DevOps Pipeline** using GitHub Actions to automate the build and deployment process:

* **Workflow:** `.github/workflows/docker_push.yml`
* **Trigger:** Activates automatically on every `git push` to the `main` branch.
* **Action:**
    1.  Checks out the latest code.
    2.  Logs in to **Docker Hub**.
    3.  Builds a new Docker image (`arth-insight:latest`).
    4.  Pushes the image to the registry, ensuring the live deployment on Render is always in sync with the repository.
  
The project employs a "Serverless Data Engineering" approach using **GitHub Actions**:

* **Workflow:** `.github/workflows/docker_market_etl.yml`
* **Schedule:** Runs strictly during **NSE Market Hours** (9:15 AM - 3:30 PM IST) to conserve resources.
* **Logic:**
    1.  Validates market status (Open/Closed/Holiday).
    2.  Fetches the Nifty 500 universe.
    3.  Performs batch processing to update fundamental metrics in BigQuery.
    4.  Ensures the warehouse remains fresh without manual intervention.

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

1. **Cold Start Latency:** Hosted on **Render Free Tier**. The application "sleeps" after inactivity, causing a 30-50 second delay on the very first request. Subsequent requests are instant.
2. **API Rate Limits:** Relies on public financial APIs. High-frequency automated scraping may lead to temporary IP throttling.
3. **Data Granularity:** Intraday charts are limited to 1-minute intervals; tick-by-tick data is reserved for enterprise-grade feeds.

---

## 🔮 Future Roadmap

* **v2.0:** Integrate Large Language Models (LLMs) to summarize quarterly earnings call transcripts into "Bull vs. Bear" bullet points.
* **v2.1:** User Authentication system allowing persistent portfolios and price alert configurations via Email/WhatsApp.
* **v2.2:** Implementation of a Technical Screener (e.g., "Show all stocks with RSI < 30").

---

**Author:** Narendra Bhandari

*Full Stack Developer & Data Engineer*
