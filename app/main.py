from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import strategies, models, data, backtest, sentiment
import os

app = FastAPI(title="TradeX API")

# ---------------------------------------------------------------------------
# CORS – explicit origins + regex to cover all Vercel preview deployments
# Add EXTRA_ORIGINS env var on Railway for any additional domains:
#   e.g.  EXTRA_ORIGINS=https://my-custom-domain.com
# ---------------------------------------------------------------------------

_base_origins = [
    "https://tradex-rho-seven.vercel.app",   # production Vercel deployment
    "http://localhost:5173",                   # Vite dev server
    "http://localhost:3000",                   # alternative local dev
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

_extra = os.getenv("EXTRA_ORIGINS", "")
if _extra:
    _base_origins += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_base_origins,
    # Covers ALL Vercel preview URLs: tradex-rho-seven-git-*.vercel.app, etc.
    allow_origin_regex=r"https://tradex-.*\.vercel\.app",
    allow_credentials=True,
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