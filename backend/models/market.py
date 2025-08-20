from sqlalchemy import Column, Integer, String, Float, BigInteger, UniqueConstraint, Index
from ..data.db import Base


class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timeframe = Column(String(10), index=True, nullable=False)
    ts = Column(BigInteger, index=True, nullable=False)  # milliseconds
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "ts", name="uq_symbol_timeframe_ts"),
        Index("idx_symbol_timeframe_ts", "symbol", "timeframe", "ts"),
    )
