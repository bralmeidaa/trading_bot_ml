"""Shared pytest fixtures for the trading bot test suite."""
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Use in-memory SQLite for all DB tests
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def _make_ohlcv(n: int = 500, seed: int = 42, trend: float = 0.0001) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with controllable trend."""
    np.random.seed(seed)
    close = 15.0 * np.exp(np.cumsum(np.random.normal(trend, 0.003, n)))
    noise = np.abs(np.random.normal(0, 0.002, n))
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + np.random.normal(0, 0.001, n))
    volume = np.random.uniform(100_000, 500_000, n)
    now = datetime.now()
    timestamps = [
        int((now - timedelta(minutes=5 * (n - i))).timestamp() * 1000)
        for i in range(n)
    ]
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high,
         "low": low, "close": close, "volume": volume}
    )


@pytest.fixture(scope="session")
def sample_ohlcv():
    """500-bar OHLCV with slight upward trend."""
    return _make_ohlcv(500)


@pytest.fixture(scope="session")
def large_ohlcv():
    """2000-bar OHLCV for initialization tests."""
    return _make_ohlcv(2000)


@pytest.fixture
def signal_gen():
    from production_trading_system import OptimizedSignalGenerator
    return OptimizedSignalGenerator("LINK/USDT", "5m")


@pytest.fixture
def trained_signal_gen(large_ohlcv):
    from production_trading_system import OptimizedSignalGenerator
    gen = OptimizedSignalGenerator("LINK/USDT", "5m")
    gen.initialize_from_history(large_ohlcv.copy())
    return gen


@pytest.fixture
def risk_manager():
    from backend.core.risk import AdvancedRiskManager, RiskParams
    return AdvancedRiskManager(RiskParams())


@pytest.fixture
def fresh_db():
    """Create a fresh in-memory DB and return the session factory."""
    from backend.persistence.database import init_db, engine, SessionLocal
    from backend.persistence.models import Base
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield SessionLocal
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def api_client():
    """FastAPI test client with no running trading system."""
    from fastapi.testclient import TestClient
    from api_server import app
    with TestClient(app) as client:
        yield client
