import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from Trading_Algo.app import (
    # get_ticker_symbol,
    get_mean_reversion_signals,
    get_stock_recommendations,
)

app = FastAPI()

class MeanReversionRequest(BaseModel):
    symbol: str  # Stock symbol of the company 
    index: str  # Market Index (e.g., S&P 500, NSE)
    lookback_days: int
    investment_amount: float

class StockRecommendationRequest(BaseModel):
    search_term: str
    market_index: str

@app.get("/")
def root():
    return {"message": "UpValue Future Prediction API Testing"}

@app.post("/mean-reversion-signals")
def mean_reversion_signals(request: MeanReversionRequest):
    try:
        result = get_mean_reversion_signals(
            symbol=request.symbol,
            index=request.index,
            lookback_days=request.lookback_days,
            investment_amount=request.investment_amount,
        )
        print(result)
        
        if not result:
            raise HTTPException(status_code=404, detail="Mean reversion signals not found")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stock-recommendations")
def stock_recommendations(request: StockRecommendationRequest):
    result = get_stock_recommendations(
        search_term=request.search_term,
        market_index=request.market_index,
    )
    return result

@app.get("/markets")
def markets():
    return get_markets()

@app.get("/default-stocks/{market}")
def default_stocks(market: str):
    result = get_default_stocks(market)
    if not result:
        raise HTTPException(status_code=404, detail="Market not found")
    return result

def main():
    print("Running FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
