# 🚀 TradeX Backend

FastAPI backend for TradeX covering 📊 market data, 🤖 ML/DL models, 📈 backtesting, and 💬 Reddit sentiment (FinBERT).

---

## 🧱 Tech Stack
- ⚡ FastAPI + Uvicorn  
- 🐘 PostgreSQL + asyncpg  
- 🧩 SQLAlchemy (async)  
- 🧠 FinBERT (Hugging Face)  
- 📊 Pandas  
- 👽 Reddit API (PRAW)

---

## 📁 Structure
app/
 ├── api/routes/      # Endpoints  
 ├── models/          # DB models  
 ├── schemas/         # Pydantic schemas  
 ├── services/        # Business logic  
 ├── db/              # DB session  
 └── core/config.py   # Settings  

---

## ▶️ Run Server

uvicorn app.main:app --reload  

Docs:  
- Swagger 👉 http://localhost:8000/docs  
- ReDoc 👉 http://localhost:8000/redoc  

---

## 🔌 API Overview

### 📊 Data
- Fetch & store OHLCV  
- Resample timeframes  

### 🧠 Strategies
- List & filter strategies  
- View full configs  

### 🤖 Models
- ML/DL results  
- Metrics (PnL, Sharpe, etc.)  

### 📈 Backtest
- Run strategy simulations  
- Get trades, PnL, stats  

### 💬 Sentiment
- Reddit scraping  
- FinBERT analysis  
- Hourly + overall sentiment  

---

## 🧠 Key Ideas
- ⚡ Async-first (non-blocking backend)  
- 🧩 Dynamic tables (no migrations needed)  
- 🗂️ Schema-based DB organization  
- 🔄 Thread pool for heavy tasks  

---

## 💡 Summary

TradeX Backend is a scalable, async trading engine that combines **market data + ML + sentiment analysis** into a single API-driven system 🚀
