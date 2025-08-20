from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ..backtesting.engine import BacktestResult


class BaseStrategy(ABC):
    """Strategy interface for backtesting on a candles DataFrame.

    Expects df columns: ['ts','open','high','low','close','volume'] plus indicators.
    """

    @abstractmethod
    def run(self, df: pd.DataFrame) -> "BacktestResult":
        ...
