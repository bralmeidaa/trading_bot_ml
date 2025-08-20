# Trading Bot with Machine Learning

A robust automated trading application for short-term operations (day trading/scalping) that exploits cryptocurrency volatility using machine learning techniques.

## Features

- **Multi-Bot Management**: Run up to 5+ bots simultaneously on different currency pairs
- **Machine Learning Integration**: Random Forest, LSTM, and Ensemble models for predictions
- **Real-time Data**: Binance API integration for live market data
- **Risk Management**: Dynamic position sizing, stop-loss, take-profit
- **Technical Indicators**: RSI, MACD, Bollinger Bands, EMA, ATR
- **Paper Trading**: Safe simulation mode for strategy testing
- **Real-time Dashboard**: React.js frontend with live charts and bot controls
- **Advanced Analytics**: Performance metrics, backtesting, ML advisor

## Architecture

```
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   ├── core/               # Core trading logic
│   ├── ml/                 # Machine learning models
│   ├── data/               # Data collection and storage
│   └── utils/              # Utilities and helpers
├── frontend/               # React.js dashboard
├── database/               # Database schemas and migrations
├── docker/                 # Docker configuration
└── tests/                  # Test suites
```

## Tech Stack

- **Backend**: Python, FastAPI, WebSockets
- **Frontend**: React.js, Recharts/Plotly
- **Database**: SQLite (dev), PostgreSQL (prod)
- **ML**: scikit-learn, TensorFlow/PyTorch
- **API**: Binance API
- **Containerization**: Docker

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (Binance API keys)
4. Run database migrations
5. Start the backend: `uvicorn backend.main:app --reload`
6. Start the frontend: `npm start`

## Configuration

- Configure trading pairs in `config/trading_pairs.json`
- Set risk parameters in `config/risk_management.json`
- ML model parameters in `config/ml_config.json`

## Safety Notice

This bot is designed for educational and research purposes. Always test thoroughly in paper trading mode before using real funds. Cryptocurrency trading involves significant risk.
