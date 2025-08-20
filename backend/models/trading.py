from sqlalchemy import Column, Integer, String, Float, BigInteger, JSON
from ..data.db import Base


class TradeLog(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    side = Column(String(4), nullable=False)  # BUY/SELL
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    ts = Column(BigInteger, index=True, nullable=False)  # milliseconds
    meta = Column(JSON, nullable=True)  # extra info: confidence, indicators, etc.
