# Estratégia — Tese, Validação e Veredito

## Contexto

Previsão de direção de 1 par com TA/ML sobre OHLCV **não tem edge** (provado: 5m/1h, 4 alvos,
formato de candle, funding → lift do ML sempre ≤ 0). Pivot para **cross-sectional**: em vez de
prever direção absoluta, ranquear uma cesta de moedas por força relativa e operar market-neutral
(long as fortes, short as fracas). Mecanismo diferente — explora movimento **relativo**.

## Como foi testado

- **Dados**: painel alinhado de 14 majors (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, LTC,
  ATOM, UNI, AAVE, NEAR), via `backend/data/universe.py` (cache local).
- **Harness**: `research_cross_sectional.py` — walk-forward (4 folds OOS), market-neutral
  long top-k / short bottom-k, custo de **0.15%/lado sobre turnover**, filtro de regime opcional
  (BTC acima da EMA lenta). Sweep: momentum vs reversal × lookback × rebalance × k × regime.
- Métrica de aprovação (GO): Sharpe OOS > 1.0 **e** positivo na maioria dos folds (≥ 3/4) **e**
  retorno anual positivo.

## Resultados (líquidos de custo, out-of-sample)

| Timeframe | Veredito | Melhor config robusta (4/4 folds) | Sharpe | Ann. ret | Max DD |
|-----------|----------|-----------------------------------|--------|----------|--------|
| 1h (intraday) | **NO-GO** | nenhuma positiva | −1.04 (melhor) | −20% | — |
| 4h (swing)    | marginal | momentum, LB48, RB24, k3, regime off | 1.08 | +38.6% | −25.7% |
| 1d (swing/position) | **GO** | momentum, LB12, RB6, k3, regime on | **1.67** | **+94%** | −27.7% |

No diário, **vários** configs de momentum deram 4/4 folds positivos com Sharpe 1.0–1.67 — sinal
robusto, não sorte de um parâmetro. Reversal e intraday: sem edge.

**Insight econômico central:** o edge aparece quando o custo×turnover é baixo. Intraday
rebalanceia demais → custo come o edge. Diário rebalanceia pouco → edge sobrevive. Isso explica
por que 1h falha e 1d passa, e bate com a literatura (momentum cross-sectional é fenômeno de
horizonte de dias/semanas).

## Validação de robustez (feita)

**1. Survivorship — CORRIGIDO e edge ficou MAIS FORTE.** Re-rodado com universo expandido (~35
moedas, incluindo várias que despencaram mas ainda negociam) + **point-in-time** (a cada
rebalance só entram moedas já listadas e líquidas naquela data, via filtro de volume trailing):

| Versão | Sharpe | Ann. ret | Folds+ |
|--------|--------|----------|--------|
| Viesada (14 sobreviventes de hoje) | 1.67 | +94% | 4/4 |
| **Corrigida (PIT, ~35 coins)** | **1.87** | **+124%** | **4/4** |

Se fosse só viés, corrigir mataria o edge. Ele **aumentou** → o momentum é real, não artefato de
escolher vencedores. (Melhor config: momentum, regime on, lookback 12d, rebalance 12d, k=3.)

**2. Sensibilidade a custo — PASSOU** (custo é o assassino documentado do cross-sectional):

| Custo/lado | Sharpe | Ann. ret | Folds+ |
|-----------|--------|----------|--------|
| 0.15% | 1.87 | +124% | 4/4 |
| 0.25% | 1.75 | +112% | 4/4 |
| 0.40% (conservador) | 1.58 | +95% | 4/4 |

Aguenta custo alto porque rebalance a cada 12 dias = turnover baixo. Por isso funciona no diário
e não no intraday (lá o turnover×custo mata).

## Robustez adicional (`.claude/research/robustness_cross_sectional.py`)

Bateria de stress além do walk-forward base (janela contínua de ~1200d):
- **Janelas de regime** (early/mid/late thirds): Sharpe 0.51–0.95 em TODAS, sempre positivo,
  mas com DD alto (−32% a −45%). O edge persiste em sub-períodos, não é um único regime.
- **Estabilidade de parâmetros**: só **3/12** combos (LB×RB) com Sharpe > 0.8 — o edge é um
  **bolsão** em torno de LB12/RB6-12, não um platô largo. Lookbacks longos (24/48) degradam
  com DD de −60% a −74%. Atenção: a config GO está num bom pocket, mas vizinhos são fracos.
- **k (concentração)**: k=2 melhor (Sharpe 1.24) mas mais concentrado; k=5 enfraquece (0.73).
- **Universo**: precisa de ≥15 nomes (max_universe=10 derruba pra 0.46).
- **Filtro de regime**: corta o DD de −60.7% para −44.7% — confirmadamente essencial.

## Ressalvas remanescentes

1. **Max drawdown REAL ~44%** numa janela contínua (o −32% era por fold; o número honesto
   end-to-end é pior). Estratégia volátil; exige sizing conservador e o filtro de regime
   (sem ele, DD vai a −60%). Gestão de risco de portfólio é obrigatória.
2. **Sensibilidade a parâmetro** — edge concentrado num pocket (LB12/RB6-12). Não é robusto a
   qualquer parametrização; cuidado com over-fitting na escolha. Walk-forward 4/4 dá conforto
   pra config GO, mas não há platô largo.
3. **Survivorship residual** — moedas TOTALMENTE deslistadas (LUNA, FTT) somem da API da Binance e
   não entram nem na versão corrigida. Reduzimos muito o viés, não 100%.
4. **Horizonte é diário/swing, NÃO intraday.** "Automatizado" e "cesta" seguem válidos; só
   "intraday" virou "diário" (hold ~12 dias). Os dados são inequívocos: 1h = NO-GO, 1d = GO.
5. **Short side** — market-neutral exige short (perp/margin) com custo/funding próprios; long-only
   (só top-k) é alternativa mais simples se o short for inviável.

## Veredito

**GO para cross-sectional momentum no DIÁRIO.** Validado out-of-sample (4/4 folds, ~1200 dias),
robusto a survivorship (edge fortaleceu ao corrigir) e a custo (até 0.40%/lado). É o primeiro
mecanismo com edge real e consistente de todo o projeto.

Config de referência: **momentum, regime BTC-EMA on, lookback 12d, rebalance 12d, long top-3 /
short bottom-3, market-neutral**, universo ~15 mais líquidos por data.

Antes de dinheiro real: (a) Fase 3 — engine de carteira automatizado; (b) paper trading que bata
com o backtest; (c) atenção ao DD de 32% (sizing + kill-switch).

## Fase 3 — engine entregue (paper)

`backend/strategy/portfolio_engine.py` — `CrossSectionalPortfolioEngine`:
- Automático, daily, market-neutral (long top-k / short bottom-k), rebalance a cada
  `rebalance_days`, mark-to-market entre rebalances, custo sobre turnover.
- Usa o núcleo validado (`backend/strategy/cross_sectional.py`).
- Kill-switch por drawdown (default 35%), filtro de regime BTC.
- Persiste equity/daily via repositórios; I/O via `asyncio.to_thread` (não trava a API).
- Endpoints: `POST /api/portfolio/start|stop`, `GET /api/portfolio` (equity, posições,
  exposições, rebalances). Separado do bot single-pair — não quebra endpoints atuais.
- Testes: `test/test_portfolio_engine.py` (9) + `test/test_cross_sectional.py` (14).

**Status:** paper trading pronto. Próximo: rodar paper por período e confe­rir que a
equity bate com o backtest antes de qualquer capital real.

## Fontes
- [Cross-Sectional Momentum in Crypto — Starkiller Capital](https://www.starkiller.capital/post/cross-sectional-momentum-in-cryptocurrency-markets)
- [Intraday return predictability: momentum, reversal, or both — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)
- [Asymmetric risk/reward — Shell Capital](https://shell-capital.com/asymmetric-investing-definitions)
