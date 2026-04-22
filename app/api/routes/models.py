from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_models():
    return [
        {
            "model_name": "LSTM Model",
            "long_trades": 50,
            "short_trades": 30,
            "win_trades": 60,
            "loss_trades": 20,
            "win_rate": 75,
            "loss_rate": 25,
            "max_drawdown": 10,
            "max_consecutive_wins": 8,
            "max_consecutive_losses": 3
        }
    ]