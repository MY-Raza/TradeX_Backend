from fastapi import FastAPI
from app.api.routes import strategies, models, backtest, data, sentiment

app = FastAPI(title="TradeX API")

app.include_router(strategies.router, prefix="/strategies")
app.include_router(models.router, prefix="/models")
# app.include_router(backtest.router, prefix="/backtest")
# app.include_router(data.router, prefix="/data")
# app.include_router(sentiment.router, prefix="/sentiment")

@app.get("/")
def root():
    return {"message": "TradeX Backend Running"}