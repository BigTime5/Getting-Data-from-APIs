
# 📈 Stock Volatility Forecasting API  
**Empowering Smarter Investments with Real-Time Volatility Predictions**

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green?logo=fastapi)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Docker](https://img.shields.io/badge/Docker-Supported-blue?logo=docker)
![CI](https://github.com/yourusername/volatility-api/actions/workflows/ci.yml/badge.svg)

---

## 🚀 Project Overview

This API forecasts daily **stock volatility** using a **GARCH(1,1)** model, tailored for high-volatility equities like `AMZN` and `WMT`. Built with **FastAPI** and powered by a **local SQLite database**, the system enables data-driven decisions in real-time — helping investors hedge, allocate, and optimize with confidence.

> ⚠️ Volatility forecasts improve **risk-adjusted returns by up to 10%**, as validated by walk-forward backtests.

---

## 🎯 Why It Matters

Investors constantly face uncertainty. Without accurate volatility forecasts, it's nearly impossible to:

- Time market entries/exits
- Adjust position sizes
- Avoid unexpected drawdowns

### ✅ This project solves that by delivering:
- **Real-time predictions** via RESTful API
- **Clustering-aware volatility modeling**
- **Reusable, automated pipeline** — from data wrangling to forecasting
- **Zero redundant API calls**, thanks to a built-in SQLite cache

---

## 🌟 Key Features

| Feature               | Description |
|-----------------------|-------------|
| 📥 **Data Ingestion** | Pulls historical stock data (`AMZN`, `WMT`) from Alpha Vantage and stores it in `stocks.sqlite`. |
| 📊 **Exploratory Analysis** | Visualizes trends in prices, returns, and volatility using `matplotlib`. |
| 🧠 **GARCH(1,1) Modeling** | Captures volatility clustering using ACF/PACF analysis on squared returns. |
| ⚡ **FastAPI Endpoints** | `/fit` trains the model; `/predict` returns n-day volatility forecasts as JSON. |
| 🧪 **Walk-Forward Validation** | Confirms model accuracy with rolling forecasts and residual diagnostics. |
| 💾 **Cost-Efficient Storage** | Reduces Alpha Vantage API calls, staying within free-tier limits. |

---

## 📡 API Usage

### 🔧 `/fit` – Train the Model
Trains a GARCH(1,1) model for the specified ticker and caches results locally.

**Request**:
```bash
POST http://localhost:8008/fit?ticker=AMZN
```

---

### 🔮 `/predict` – Get Forecast
Returns `n_days` of predicted daily volatility in percentage terms.

**Request**:
```bash
POST http://localhost:8008/predict?ticker=AMZN&n_days=5
```

**Sample Response**:
```json
{
  "ticker": "AMZN",
  "n_days": 5,
  "success": true,
  "forecast": {
    "2025-05-16T00:00:00": 3.3285,
    "2025-05-19T00:00:00": 3.752,
    "2025-05-20T00:00:00": 4.1323,
    "2025-05-21T00:00:00": 4.4804,
    "2025-05-22T00:00:00": 4.8034
  },
  "message": "Forecast generated for AMZN over 5 days."
}
```

---

## 🧠 Model Validation

- **Volatility Bands**: ±2 SD bands show good alignment with actual returns.
- **Residual Diagnostics**: Standardized residuals are near-normal, showing no autocorrelation.
- **Annualized Volatility**: `AMZN` (44.71%) vs `WMT` (30.23%) justifies different investment strategies.

---

## 📦 Project Structure

```
📁 src/
│   ├── main.py         # FastAPI entry point
│   ├── model.py        # GARCH model logic
│   ├── data.py         # Data ingestion and SQLite storage
│   ├── config.py       # Configs & constants
│
📁 notebooks/
│   └── deploy.ipynb    # Full pipeline from data wrangling to validation
│
📁 db/
│   └── stocks.sqlite   # Cached time series data
```

---

## 🔮 Roadmap

- [ ] **Cloud Deployment** on Render using Docker
- [ ] **Ticker Expansion**: Add `NFLX`, `GOOGL`, etc.
- [ ] **Interactive Dashboard**: Visualize forecasts with Plotly or Tableau
- [ ] **Model Upgrades**: Explore EGARCH or ensemble volatility models
- [ ] **Robust Testing**: Add edge case tests for missing/invalid data

---

## 🔐 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 🙌 Acknowledgments

- **Alpha Vantage** – Stock market data API
- **arch** – GARCH modeling library
- **FastAPI** – Lightning-fast web framework
- **matplotlib/pandas** – Data wrangling & visualization tools
