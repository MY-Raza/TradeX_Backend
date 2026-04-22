from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_strategies():
    return [
        {
            "name": "RSI Strategy",
            "symbol": "BTCUSDT",
            "time_horizon": "1h",
            "indicators": ["RSI"],
            "patterns": ["Overbought/Oversold"]
        }
    ]