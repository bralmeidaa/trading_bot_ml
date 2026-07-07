# -*- coding: utf-8 -*-
"""
Microstructure research harness — analyzes the live-collected order book data.

  --sync   scp the order book CSVs from the VM to .claude/research/_obcache/
  (then)   decay curve (signal vs forward return over horizons) + threshold
           strategy (trade only extreme imbalance) with GROSS and NET-of-cost.

Answers: is there a tradeable order-flow signal at the 60s collection cadence?

Run:
  python .claude/research/research_microstructure.py --sync
  python .claude/research/research_microstructure.py            # use cached
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import _bootstrap  # noqa: F401

import os
import glob
import argparse
import subprocess

import numpy as np
import pandas as pd

from backend.data.microstructure import decay_correlations, forward_return, threshold_backtest

CACHE = os.path.join(os.path.dirname(__file__), "_obcache")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
# VM access (see memory ssh-vm-access)
SSH_KEY = os.getenv("OB_SSH_KEY",
                    r"C:\Users\Bruno.Almeida\OneDrive\Estudos\DevOps\ubt-vm-01\.ssh\ubt-vm-01-2024-06-26.key")
VM = os.getenv("OB_VM", "ubuntu@130.61.28.178")
REMOTE_DIR = "/opt/trading-bot-data/orderbook_data"

HORIZONS = [1, 2, 4, 8, 16]                 # snapshots (~80s each) → ~1.3min..21min
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "XRP/USDT"]
QUANTILES = [0.80, 0.90, 0.95]
COSTS_BPS = [2, 3, 4]


def sync():
    os.makedirs(CACHE, exist_ok=True)
    key = SSH_KEY.replace("\\", "/")
    if key[1:3] == ":/":
        key = "/" + key[0].lower() + key[2:]      # C:/... → /c/...
    cmd = ["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null",
           f"{VM}:{REMOTE_DIR}/*.csv", CACHE + "/"]
    print("Syncing order book CSVs from VM...")
    subprocess.run(cmd, check=False)


def load() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(CACHE, "orderbook_*.csv")))
    if not files:
        print(f"No CSVs in {CACHE}. Run with --sync first.")
        sys.exit(1)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates(["timestamp", "symbol"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    if args.sync:
        sync()

    df = load()
    span_d = (df.timestamp.max() - df.timestamp.min()) / 1000 / 86400
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# Microestrutura — order book coletado\n")
    out(f"{len(df):,} linhas · {df.symbol.nunique()} símbolos · {span_d:.1f} dias · cadência ~60s\n")

    # ---- decay curve ----
    out("## 1. Curva de decaimento — corr(sinal(t), retorno futuro)\n")
    out("Horizonte em snapshots (~80s cada). imb=imbalance_top20, mp=microprice_dev.\n")
    hdr = "| símbolo | sinal | " + " | ".join(f"h{h}" for h in HORIZONS) + " |"
    out(hdr); out("|" + "---|" * (len(HORIZONS) + 2))
    for sym in SYMBOLS:
        s = df[df.symbol == sym].sort_values("timestamp").reset_index(drop=True)
        if len(s) < 500:
            continue
        for sig_col, lbl in [("imbalance_top20", "imb"), ("microprice_dev", "mp")]:
            c = decay_correlations(s, HORIZONS, sig_col)
            row = f"| {sym} | {lbl} | " + " | ".join(f"{c[h]:+.3f}" for h in HORIZONS) + " |"
            out(row)

    # ---- threshold strategy (h=1, the strongest horizon) ----
    out("\n## 2. Estratégia de threshold (operar só imbalance extremo), horizonte h1 (~80s)\n")
    out("Líquido = bruto − custo (round-trip). Projeção diária = net_bps × trades/dia.\n")
    out("| símbolo | quantil | n | WR | bruto(bps) | net@2 | net@3 | net@4 | /dia@3bps |")
    out("|---|---|---|---|---|---|---|---|---|")
    best = []
    for sym in SYMBOLS:
        s = df[df.symbol == sym].sort_values("timestamp").reset_index(drop=True)
        if len(s) < 500:
            continue
        days = (s.timestamp.max() - s.timestamp.min()) / 1000 / 86400
        imb = s["imbalance_top20"]
        fwd = forward_return(s["mid"], 1)
        for q in QUANTILES:
            r2 = threshold_backtest(imb, fwd, q, 2)
            r3 = threshold_backtest(imb, fwd, q, 3)
            r4 = threshold_backtest(imb, fwd, q, 4)
            if not r3:
                continue
            per_day = r3["net_bps"] * (r3["n"] / max(days, 1))
            out(f"| {sym} | p{int(q*100)} | {r3['n']} | {r3['win_rate']:.0%} | "
                f"{r3['gross_bps']:+.2f} | {r2['net_bps']:+.2f} | {r3['net_bps']:+.2f} | "
                f"{r4['net_bps']:+.2f} | {per_day:+.2f}bps |")
            if r3["net_bps"] > 0:
                best.append((sym, q, r3, per_day))

    # ---- verdict ----
    out("\n## Veredito\n")
    if best:
        out("Configs com NET positivo a 3bps de custo:")
        for sym, q, r, pd_ in sorted(best, key=lambda x: -x[3]):
            out(f"- {sym} p{int(q*100)}: net {r['net_bps']:+.2f}bps/trade, "
                f"{pd_:+.2f}bps/dia, WR {r['win_rate']:.0%}, n={r['n']}")
        out("\n→ Há sinal que paga o custo em ALGUM threshold — investigar HF/execução com cuidado.")
    else:
        out("Nenhum threshold dá NET positivo a 3bps de custo. O sinal (bruto ~0.4-0.7bps) é")
        out("real mas < custo mesmo filtrando extremos → cadência de 60s não basta; edge de")
        out("order flow é sub-segundo (HF/latência). Confirma o primeiro olhar.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "microstructure_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nRelatório salvo em {path}")


if __name__ == "__main__":
    main()
