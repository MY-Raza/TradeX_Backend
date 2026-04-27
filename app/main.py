from fastapi import FastAPI
from app.api.routes import strategies, models,data,backtest,sentiment 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TradeX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tradex-rho-seven.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(strategies.router)
app.include_router(models.router)
app.include_router(data.router)
app.include_router(backtest.router)
app.include_router(sentiment.router)
@app.get("/")
def root():
    return {"message": "TradeX Backend Running"}