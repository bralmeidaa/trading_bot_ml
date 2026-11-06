"""
Advanced risk management: Kelly Criterion + volatility-adjusted position sizing.
Used by ProductionTradingSystem to replace the fixed-risk sizing formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RiskParams:
    # Kelly Criterion settings
    kelly_fraction: float = 0.25    # conservative 25% of full Kelly
    kelly_lookback: int = 50        # trades to look back for win/loss stats
    kelly_min_trades: int = 10      # minimum trades before Kelly activates

    # Position size bounds (as % of allocated capital)
    base_risk_pct: float = 0.01     # 1.0% base risk per trade
    max_risk_pct: float = 0.03      # 3.0% ceiling
    min_risk_pct: float = 0.001     # 0.1% floor

    # Volatility targeting
    target_volatility: float = 0.02   # 2% daily vol target
    volatility_lookback: int = 20     # bars for vol estimate

    # Drawdown protection
    max_drawdown_pct: float = 0.10    # reduce sizing when approaching this

    # Execution cost estimate for net-return calculation
    commission_pct: float = 0.001     # 0.1% per side
    slippage_pct: float = 0.0005      # 0.05% per side


class AdvancedRiskManager:
    """
    Calculates position size using a three-layer model:
      1. Kelly Criterion (from recent trade history)
      2. Volatility scaling (inverse: high vol → smaller size)
      3. Confidence scaling (0.5x–1.5x based on signal strength)

    Falls back to `base_risk_pct` until `kelly_min_trades` results are recorded.
    """

    def __init__(self, params: RiskParams | None = None):
        self.params = params or RiskParams()
        self._results: List[float] = []   # rolling pnl_pct per trade

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, pnl_pct: float) -> None:
        """Record a closed trade result. Must be called after every exit."""
        self._results.append(pnl_pct)
        max_history = self.params.kelly_lookback * 4
        if len(self._results) > max_history:
            self._results = self._results[-max_history:]

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 0.7,
        current_equity: float | None = None,
        peak_equity: float | None = None,
        returns: pd.Series | None = None,
    ) -> float:
        """Return the number of units to trade.

        Args:
            capital:        Allocated capital for this bot (total * allocation_pct).
            entry_price:    Estimated execution price.
            stop_loss:      Stop-loss price.
            confidence:     Signal confidence in [0, 1].
            current_equity: Current portfolio equity (for drawdown protection).
            peak_equity:    Highest historical equity (for drawdown calculation).
            returns:        Recent price returns series (for volatility scaling).
        """
        risk_pct = self._kelly_risk_pct()

        # Layer 2: volatility scaling
        if returns is not None and len(returns) >= self.params.volatility_lookback:
            risk_pct *= self._volatility_multiplier(returns)

        # Layer 3: confidence scaling (0.5x at conf=0, 1.5x at conf=1)
        risk_pct *= 0.5 + confidence

        # Layer 4: drawdown protection
        if current_equity is not None and peak_equity is not None and peak_equity > 0:
            dd = (peak_equity - current_equity) / peak_equity
            if dd > 0:
                dd_factor = max(0.1, 1.0 - dd / max(self.params.max_drawdown_pct, 1e-9))
                risk_pct *= dd_factor

        risk_pct = float(np.clip(risk_pct, self.params.min_risk_pct, self.params.max_risk_pct))

        stop_distance = abs(entry_price - stop_loss) / entry_price
        if stop_distance <= 0:
            return 0.0

        risk_amount = capital * risk_pct
        position_value = risk_amount / stop_distance
        return position_value / entry_price  # units

    def apply_costs(self, price: float, side: str) -> float:
        """Return the effective execution price after commission + slippage."""
        cost = self.params.commission_pct + self.params.slippage_pct
        return price * (1 + cost) if side == "buy" else price * (1 - cost)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kelly_risk_pct(self) -> float:
        results = self._results[-self.params.kelly_lookback:]
        if len(results) < self.params.kelly_min_trades:
            return self.params.base_risk_pct

        wins = [r for r in results if r > 0]
        losses = [r for r in results if r <= 0]

        if not wins or not losses:
            return self.params.base_risk_pct

        win_rate = len(wins) / len(results)
        avg_win = float(np.mean(wins))
        avg_loss = abs(float(np.mean(losses)))

        if avg_loss == 0:
            return self.params.base_risk_pct

        b = avg_win / avg_loss
        raw_kelly = (b * win_rate - (1 - win_rate)) / b
        conservative = max(0.0, raw_kelly) * self.params.kelly_fraction
        return float(np.clip(conservative, self.params.min_risk_pct, self.params.max_risk_pct))

    def _volatility_multiplier(self, returns: pd.Series) -> float:
        recent = returns.iloc[-self.params.volatility_lookback:]
        vol = recent.std()
        if vol <= 0:
            return 1.0
        ratio = self.params.target_volatility / vol
        return float(np.clip(ratio, 0.1, 3.0))
