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

## Ressalvas honestas (antes de confiar/automatizar)

1. **Survivorship bias** — o universo são os majors de HOJE, que sobreviveram. Moedas que
   quebraram não estão no painel. Isso infla retornos de momentum (vencedores que seguiram
   vencendo). É a ressalva mais séria; idealmente validar com universo point-in-time.
2. **Dependência de regime** — momentum em cripto brilha em mercados em tendência (2021-2024 teve
   bull runs fortes). O filtro de regime (BTC EMA) ajuda e reduz drawdown, mas chop/bear vão doer.
   Max DD ~28% já é alto.
3. **Horizonte** — o edge é **diário/swing**, NÃO intraday. "Totalmente automatizado" e "cesta de
   majors" seguem válidos; só o "intraday" precisa virar "diário" — os dados são inequívocos.
4. **Short side** — market-neutral exige vender a descoberto (perp/margin), com custo e
   funding próprios; a versão long-only (top-k) é alternativa mais simples.

## Veredito

**GO condicional para cross-sectional momentum no DIÁRIO.** É o primeiro mecanismo com edge OOS
consistente de todo o projeto. Antes de automatizar com dinheiro: (a) endereçar survivorship
(universo point-in-time ou aceitar a ressalva), (b) validar em paper trading que bata com o
backtest, (c) confirmar comportamento em janela de bear/chop.

## Fontes
- [Cross-Sectional Momentum in Crypto — Starkiller Capital](https://www.starkiller.capital/post/cross-sectional-momentum-in-cryptocurrency-markets)
- [Intraday return predictability: momentum, reversal, or both — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)
- [Asymmetric risk/reward — Shell Capital](https://shell-capital.com/asymmetric-investing-definitions)
