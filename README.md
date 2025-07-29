# Mean Reversion Strategy Analyzer

A web application that analyzes stocks using a mean reversion strategy optimized for day trading. The application provides buy/sell signals based on price deviations from moving averages, using an API to be integrated with other Applications

## Features

- Real-time stock analysis using mean reversion strategy
- Support for multiple markets (NSE, BSE, US)
- Interactive charts and visualizations
- Detailed trading signals and recommendations
- Risk management indicators (Stop Loss, Take Profit)

## API Endpoint checking

For Mac Users:
```
cd Trading_API
python3 -m .venv venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Tech Stack

- Frontend: StreamLit
- Backend: FastAPI (Python)
- Data Analysis: Pandas, NumPy
- Stock Data: Yahoo Finance API

## Supported Markets:
KL - malaysian_market


## Project Structure

```
mean-reversion-analyzer/
├── Trading_Alog/               # Trading Algo testing dashboard
│   ├── app.py
│   └── requirements.txt         # Algo dependencies
│   └── rest...                  

├── Trading_API/                 # Rest API for Interaction 
│   ├── main.py                  # API endpoints
│   └── requirements.txt  # Backend dependencies
└── README.md            # Project documentation
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mean-reversion-analyzer.git
   cd mean-reversion-analyzer
   ```

2. Set up the backend:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```bash
   cd backend
   uvicorn api:app --reload --port 8000
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Select your market (NSE, BSE, or US)
2. Enter stock symbols (one per line)
3. Adjust the lookback period (10-60 days recommended)
4. Set your investment amount per stock
5. Click "Analyze Stocks" to get results

## Deployment

The application can be deployed to Vercel (frontend) and any Python hosting service (backend). Make sure to set up the appropriate environment variables and CORS settings.

## License

MIT License
