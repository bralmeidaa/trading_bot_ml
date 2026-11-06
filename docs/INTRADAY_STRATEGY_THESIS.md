# Intraday — ORB+VWAP+sessão: tese, validação e veredito

## Contexto

O usuário desafiou a premissa "competimos com baixa latência". Correto: **latência só decide em
HFT** (market making, arbitragem, scalping sub-segundo). O daytrader humano (minutos-horas, poucos
trades/dia) não compete em velocidade. Então testamos a hipótese certa: **codificar uma estratégia
de trader real (ORB + VWAP + tendência multi-TF + timing de sessão) e usar IA como meta-filtro**,
validando por **setup** (não por barra), com **lucro concreto reportado** (semanal/mensal em % e $).

Base de evidência: efeitos de hora-do-dia documentados (pico 16-17 UTC), sessão NY mais confiável,
ORB+VWAP com win rate 40-55% na literatura prática. Tudo backtestável de OHLCV.

## O que foi construído (infra sólida e testada)

- `backend/data/intraday_features.py` — VWAP ancorado em sessão, opening range, flags de
  sessão/hora, tendência multi-TF, estrutura (swings), ATR, volume. **Causal (sem look-ahead).**
- `backend/strategy/intraday_setups.py` — setups ORB+VWAP+sessão **explícitos** com stop/alvo de
  R:R assimétrico + simulador triple-barrier (custo em R).
- `.claude/research/research_intraday.py` — gera setups → simula → **meta-filtro IA walk-forward**
  → reporta expectativa E **lucro semanal/mensal em $**, com consistência por fold e sweep.
- Testes: `test/test_intraday_features.py` (7) + `test/test_intraday_setups.py` (7).

## A lição central: o custo em R mata o intraday

A primeira rodada parecia promissora (+5%/mês BTC 5m) — era **miragem por um bug**: a expectativa
usava o R **bruto** (níveis de preço), ignorando custo. Corrigido para **custo em termos de R**:

> Com sizing por risco fixo, o tamanho da posição ∝ 1/distância-do-stop. Stop apertado (que o
> intraday exige, ~1×ATR ≈ 0.1-0.3% no 5m) faz o custo de ida-e-volta ser uma **fração enorme do
> R arriscado**: `cost_R = 2·custo·preço / |entrada−stop|`. A 0.05%/lado com stop 1×ATR isso é
> ~0.6R **por trade**. O edge bruto de +0.20R simplesmente não sobrevive.

## Resultados (180d, BTC+ETH, walk-forward, líquido de custo)

| Config | 0.05%/lado | 0.10%/lado |
|--------|-----------|-----------|
| stop 1×ATR, rr 2 (todos os pares/TF) | **−0.16 a −0.40R, 0-1/4 folds** | −0.4 a −1.0R, 0/4 |
| stop 3×ATR, rr 1.5 | maioria negativa; **ETH 5m +0.08R 4/4** (isolado/marginal) | — |

- **1h: NO-GO** sempre (ORB é fenômeno de sessão curta).
- Stop largo reduz o custo-por-R e quase empata, mas só **ETH 5m** ficou positivo — **um
  sobrevivente isolado e minúsculo (+1.6%/mês ≈ $19) entre muitas combinações testadas**. BTC (o
  major mais líquido) não mostra. Isso é **risco de teste múltiplo**, não edge robusto.

## Veredito

**NO-GO para intraday com inputs de OHLCV.** Não por "ruído" no abstrato, mas por um mecanismo
quantificado: **custo/R explode com stop apertado**. Stop largo mitiga mas dilui o edge e só
sobrevive marginalmente num símbolo — dentro do esperado por acaso. Sem edge confiável.

## O que sobra como hipótese (decisão do usuário)

A única alavanca não esgotada é o **input mais rico que o trader real usa e o modelo não tem:
order book / fluxo (L2, desequilíbrio, absorção)** — que **não tem histórico** pra backtest. Opção:
ligar um **coletor ao vivo** (`backend/data/orderbook_collector.py`) e acumular dados próprios por
semanas/meses pra então testar microestrutura. É aposta de médio prazo, sem garantia.

Caso contrário, o intraday está **honestamente esgotado** com dados públicos de preço, e o
**cross-sectional momentum diário** segue como a estratégia de produção (essa sim com edge
validado — ver `STRATEGY_THESIS.md`).
