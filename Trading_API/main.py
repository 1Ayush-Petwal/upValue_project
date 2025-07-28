import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Trading_Algo.core import (
    get_ticker_symbol,
    get_markets,
    _get_mean_reversion_signals_,
    get_stock_recommendations,
    get_default_stocks,
    _run_analysis_
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



## def run_analysis(symbols, lookback_days=60, investment_per_stock=100000, market_index="NS"):
# class AnalysisRequest(BaseModel):
#     symbols: []
#     looback_dats: int
#     investment_per_stock: int
#     market_index: str 


@app.get("/")
def root():
    return {"message": "UpValue Future Prediction API Testing"}

@app.get("/markets")
def markets():
    return get_markets()


class TickerRequest(BaseModel):
    symbol: str
    market_index: str

### List of Symbols:
@app.post("/ticker-symbols")
def tickers(request: TickerRequest):
    return get_ticker_symbol(
        symbol = request.symbol,
        market_index = request.market_index);

@app.post("/mean-reversion-signals")
def mean_reversion_signals(request: MeanReversionRequest):
    try:
        result = _get_mean_reversion_signals_(
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


class AnalysisRequest(BaseModel):
    symbols: List[str]
    looback_dats: int
    investment_per_stock: int
    market_indx: str 

@app.post("/Analysis")
def run_analysis(request: AnalysisRequest):
    try:
        result = _run_analysis_(
            symbols=request.symbols,
            lookback_days=request.lookback_days,
            investment_per_stock=request.investment_per_stock,
            market_index=request.market_index,
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
