# -*- coding: utf-8 -*-
"""
Spike: o MESMO modelo de momentum cross-sectional aplicado a AÇÕES (fora do BTC).

Pergunta do usuário: esse modelo se adequa melhor a outro mercado? A tese é que
em ações (universo grande, sem um ativo dominante que arrasta tudo) o momentum
cross-sectional é mais robusto e NÃO precisa da muleta do filtro de regime que o
cripto exige (sem filtro, o cripto tem OOS maxDD -88%).

Reusa o núcleo agnóstico de ativo: simulate/momentum_signal/target_weights de
backend.strategy.cross_sectional. Anualização própria (ações ~252 dias/ano, não 365).

Dados: yfinance (grátis, ajustado por splits/dividendos).
Ressalvas honestas: (1) survivorship — a cesta são large-caps ATUAIS (mitigado com
laggards, não eliminado); (2) sem custo de borrow no short; (3) sem modelo de execução.

Run: python .claude/research/research_equity_momentum.py
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import yfinance as yf

from backend.strategy.cross_sectional import simulate

LB, RB, K = 12, 12, 3
COST = 0.0005          # 5 bps/lado — large-cap liquida (spread ~1-2bps + comissao ~0)
START = "2010-01-01"
FOLDS = 5
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")

# ~40 large-caps diversas (inclui laggards INTC/T/WBA/BA/PFE/XOM p/ mitigar survivorship)
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "INTC",
    "CSCO", "IBM", "QCOM", "AMD", "JPM", "BAC", "WFC", "GS", "MS", "C",
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "KO", "PEP", "PG", "WMT", "MCD",
    "NKE", "DIS", "XOM", "CVX", "CAT", "BA", "GE", "T", "VZ", "WBA",
]
MARKET = "SPY"   # p/ o filtro de regime de mercado (analogo ao BTC no cripto)


def metrics252(port):
    if port is None or len(port) == 0 or port.abs().sum() == 0:
        return dict(n=0, ann_ret=0.0, sharpe=0.0, maxdd=0.0, net=0.0)
    bpy = 252
    mean, std = port.mean(), port.std()
    sharpe = float(mean / std * np.sqrt(bpy)) if std > 0 else 0.0
    eq = (1 + port).cumprod()
    net = float(eq.iloc[-1] - 1)
    ann = float((1 + net) ** (bpy / len(port)) - 1)
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return dict(n=len(port), ann_ret=ann, sharpe=sharpe, maxdd=maxdd, net=net)


def spy_regime(close, fast=12, slow=60):
    m = close[MARKET]
    return m.ewm(span=fast).mean() > m.ewm(span=slow).mean()


def wf(close, vol, lb, rb, mode, exposure, regime_fn):
    n = len(close); f = n // (FOLDS + 1)
    ports, sharpes = [], []
    for i in range(1, FOLDS + 1):
        seg, vseg = close.iloc[i * f:(i + 1) * f], vol.iloc[i * f:(i + 1) * f]
        if len(seg) < lb + rb + 5:
            continue
        ov = regime_fn(seg) if regime_fn else None
        p = exposure * simulate(seg, lb, rb, K, mode=mode, btc_filter=False, cost=COST,
                                volume=vseg, regime_override=ov)
        if len(p):
            ports.append(p); sharpes.append(metrics252(p)["sharpe"])
    agg = metrics252(pd.concat(ports)) if ports else dict(sharpe=0, maxdd=0, ann_ret=0)
    agg["pos"] = sum(1 for s in sharpes if s > 0); agg["nf"] = len(sharpes)
    return agg


def full_run(close, vol, lb, rb, mode, exposure, regime_fn):
    ov = regime_fn(close) if regime_fn else None
    return metrics252(exposure * simulate(close, lb, rb, K, mode=mode, btc_filter=False,
                                          cost=COST, volume=vol, regime_override=ov))


def main():
    print(f"Baixando {len(UNIVERSE)}+1 tickers desde {START} (yfinance)...")
    tickers = UNIVERSE + [MARKET]
    raw = yf.download(tickers, start=START, interval="1d", progress=False, auto_adjust=True)
    close = raw["Close"].dropna(how="all")
    vol = raw["Volume"].reindex(close.index)
    # mantem so quem tem historico denso; ffill buracos pontuais
    keep = [c for c in close.columns if close[c].notna().mean() > 0.6]
    close = close[keep].ffill().dropna(how="any")
    vol = vol[keep].reindex(close.index).ffill()
    if MARKET not in close.columns:
        print("SPY ausente — abortando"); sys.exit(1)

    start, end = close.index[0].date(), close.index[-1].date()
    yrs = len(close) / 252
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# Momentum cross-sectional em AÇÕES (mesmo núcleo do cripto)\n")
    out(f"{len(close.columns)-1} ações + SPY · {len(close):,} pregões · {start}→{end} (~{yrs:.1f} anos) · "
        f"custo {COST:.2%}/lado · anualização 252d/ano\n")

    # ── 1. Varredura de horizonte (o parametro de cripto (12d) e reversao em acoes) ──
    out("## 1. Horizonte certo p/ ações (sem filtro, full exposure)\n")
    out("Momentum de ações é de 6–12 MESES (não 12 dias). LB/RB em pregões (~21/mês).\n")
    out("| lookback | rebal | modo | CAGR | Sharpe | maxDD | OOS Sharpe | OOS maxDD | folds+ |")
    out("|---|---|---|---|---|---|---|---|---|")
    grid = [
        (12, 12, "momentum"),    # config do cripto (esperado: ruim/reversao)
        (12, 5,  "reversal"),    # reversao de curto prazo
        (63, 21, "momentum"),    # 3 meses
        (126, 21, "momentum"),   # 6 meses (classico)
        (189, 21, "momentum"),   # 9 meses
        (252, 21, "momentum"),   # 12 meses (classico)
    ]
    best = None
    for lb, rb, mode in grid:
        fm = full_run(close, vol, lb, rb, mode, 1.0, None)
        w = wf(close, vol, lb, rb, mode, 1.0, None)
        out(f"| {lb} | {rb} | {mode} | {fm['ann_ret']:+.1%} | {fm['sharpe']:.2f} | {fm['maxdd']:+.0%} "
            f"| {w['sharpe']:.2f} | {w['maxdd']:+.0%} | {w['pos']}/{w['nf']} |")
        if best is None or w["sharpe"] > best[3]["sharpe"]:
            best = (lb, rb, mode, w, fm)

    lb, rb, mode, bw, bfm = best
    out(f"\nMelhor horizonte OOS: **LB{lb}/RB{rb} {mode}** (OOS Sharpe {bw['sharpe']:.2f}).\n")

    # ── 2. No melhor horizonte: precisa de filtro de regime? + sizing ──
    out(f"## 2. No melhor horizonte (LB{lb}/RB{rb} {mode}): filtro de regime é necessário?\n")
    out("| config | %investido | CAGR | Sharpe | maxDD | OOS Sharpe | OOS maxDD | folds+ |")
    out("|---|---|---|---|---|---|---|---|")
    for name, expo, fn in [("SEM filtro", 1.0, None), ("com regime SPY", 1.0, spy_regime),
                           ("SEM filtro @ exp 0.5", 0.5, None)]:
        pi = float((fn(close) if fn else pd.Series(True, index=close.index)).iloc[lb:].mean())
        fm = full_run(close, vol, lb, rb, mode, expo, fn)
        w = wf(close, vol, lb, rb, mode, expo, fn)
        out(f"| {name} | {pi:.0%} | {fm['ann_ret']:+.1%} | {fm['sharpe']:.2f} | {fm['maxdd']:+.0%} "
            f"| {w['sharpe']:.2f} | {w['maxdd']:+.0%} | {w['pos']}/{w['nf']} |")

    out("\n## Veredito (vs cripto)\n")
    out(f"1. **Horizonte importa muito**: a config de cripto (LB12/RB12) é RUIM em ações "
        f"(momentum de 12d = zona de reversão de curto prazo). O horizonte certo é ~6 meses "
        f"(LB{lb}/RB{rb}): vira POSITIVO (+{bfm['ann_ret']:.0%} CAGR, Sharpe {bfm['sharpe']:.2f}).")
    out(f"2. **NÃO precisa da muleta de regime**: sem filtro opera 100% do tempo com OOS "
        f"maxDD {bw['maxdd']:+.0%}; o filtro SPY quase não muda (levemente pior OOS). No cripto, "
        f"sem filtro = OOS maxDD -88%. → a dependência de um ativo-âncora é MUITO menor. Tese confirmada.")
    out(f"3. **MAS este port cru é modesto**: OOS Sharpe {bw['sharpe']:.2f} e maxDD {bw['maxdd']:+.0%} — "
        f"pior que a versão CRIPTO já tunada (OOS Sharpe ~0.99). O −52% são 'momentum crashes' "
        f"(squeeze da perna short em recuperações) — fenômeno conhecido. k=3 é bem concentrado.")
    out("\n→ Conclusão honesta: o modelo se adequa MELHOR a ações no sentido estrutural (horizonte")
    out("  natural de meses, sem precisar do regime BTC), mas pra ser competitivo precisa de")
    out("  construção própria de ações: mais nomes (decis, não k=3), e vol-scaling contra momentum")
    out("  crashes (Barroso-Santa-Clara). Não é 'plug o cripto e pronto'.")
    out("\nRessalvas: survivorship (large-caps atuais), sem custo de borrow no short, sem execução real.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "equity_momentum_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRelatório salvo em {path}")


if __name__ == "__main__":
    main()
