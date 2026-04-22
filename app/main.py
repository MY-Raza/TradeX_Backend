from fastapi import FastAPI
from app.api.routes import strategies, models,data,backtest  # only import what exists

app = FastAPI(title="TradeX API")

app.include_router(strategies.router)
app.include_router(models.router)
app.include_router(data.router)
app.include_router(backtest.router)
@app.get("/")
def root():
    return {"message": "TradeX Backend Running"}