from dataclasses import dataclass


@dataclass
class RiskParams:
    risk_per_trade_pct: float = 1.0  # percent of equity
    stop_loss_pct: float = 0.5       # percent below entry for SL
    take_profit_pct: float = 1.0     # percent above entry for TP


def position_size(equity: float, price: float, risk_per_trade_pct: float) -> float:
    if price <= 0:
        return 0.0
    cash = equity * (risk_per_trade_pct / 100.0)
    return cash / price


def compute_sl_tp(entry_price: float, sl_pct: float, tp_pct: float) -> tuple[float, float]:
    sl = entry_price * (1.0 - sl_pct / 100.0)
    tp = entry_price * (1.0 + tp_pct / 100.0)
    return sl, tp
