from fastapi import FastAPI
from app.api.routes import strategies  # only import what exists

app = FastAPI(title="TradeX API")

app.include_router(strategies.router)

@app.get("/")
def root():
    return {"message": "TradeX Backend Running"}