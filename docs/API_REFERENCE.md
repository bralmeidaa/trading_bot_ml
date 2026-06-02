# Trading Bot ML — API Reference

Referência completa de todos os endpoints do `api_server.py`.

> **Swagger interativo:** `https://trading.braatech.xyz/docs`
> **ReDoc:** `https://trading.braatech.xyz/redoc`

---

## Contrato de resposta

**TODAS** as respostas de sucesso seguem o envelope padrão:

```json
{ "success": true, "data": <payload> }
```

Erros lançados via `HTTPException` seguem o padrão do FastAPI:

```json
{ "detail": "mensagem de erro" }
```

Erros 404/500 não tratados retornam:

```json
{ "success": false, "error": "mensagem" }
```

> ⚠️ **Regra de ouro para o frontend:** o hook `useApi.js` espera `result.success` e usa `result.data`.
> Componentes que fazem `fetch()` cru devem desembrulhar com `json.data ?? json`.

---

## Autenticação

Atualmente **nenhuma** — a API é aberta atrás do nginx. Quando autenticação for adicionada,
será via header `Authorization: Bearer <token>` (planejado, ainda não implementado).

---

## Índice de endpoints

| Método | Path | Categoria | Descrição |
|--------|------|-----------|-----------|
| GET  | `/api/health` | System | Health check |
| GET  | `/api/status` | System | Status do sistema |
| POST | `/api/start` | System | Iniciar trading |
| POST | `/api/stop` | System | Parar trading |
| POST | `/api/emergency-stop` | System | Parada de emergência |
| GET  | `/api/metrics` | Metrics | Métricas de trading |
| GET  | `/api/performance-metrics` | Metrics | Uso de CPU/memória |
| GET  | `/api/equity` | Metrics | Curva de equity |
| GET  | `/api/daily-stats` | Metrics | P&L diário (30d) |
| GET  | `/api/bots` | Bots | Lista de bots + status |
| GET  | `/api/bots/count` | Bots | Contagem de bots |
| GET  | `/api/bots/available-symbols` | Bots | Pares suportados |
| GET  | `/api/bots/available-timeframes` | Bots | Timeframes suportados |
| GET  | `/api/bots/{bot_id}/config` | Bots | Config de um bot |
| POST | `/api/bots/{bot_id}/toggle` | Bots | Ativar/desativar bot |
| POST | `/api/bots/add` | Bots | Adicionar bot |
| PUT  | `/api/bots/update/{index}` | Bots | Atualizar bot por índice |
| DELETE | `/api/bots/remove/{index}` | Bots | Remover bot por índice |
| GET  | `/api/config` | Config | Config simples |
| POST | `/api/config` | Config | Atualizar config simples |
| GET  | `/api/config/full` | Config | Config completa (global + bots) |
| GET  | `/api/config/backups` | Config | Listar backups |
| PUT  | `/api/config/global` | Config | Atualizar campos globais |
| PUT  | `/api/config/bot/{bot_id}` | Config | Atualizar bot por id |
| POST | `/api/config/bot` | Config | Adicionar bot (alias) |
| DELETE | `/api/config/bot/{bot_id}` | Config | Remover bot por id |
| POST | `/api/config/restore/{filename}` | Config | Restaurar backup |
| GET  | `/api/logs` | Logs | Últimos 100 logs |
| GET  | `/api/logs/statistics` | Logs | Contagem por nível |
| GET  | `/api/logs/categories` | Logs | Categorias de log |
| GET  | `/api/signal-quality` | Analytics | Qualidade dos sinais |
| GET  | `/api/market-regime/{symbol}` | Analytics | Regime de mercado |
| GET  | `/api/market-sentiment/{symbol}` | Analytics | Sentimento de mercado |
| POST | `/api/backtest` | Analytics | Disparar backtest |
| GET  | `/api/backtest/results` | Analytics | Resultados do backtest |

---

## System

### `GET /api/health`
Health check. Sempre 200 se o servidor está no ar.
```json
{ "success": true, "data": { "status": "healthy", "timestamp": "2025-06-01T12:00:00", "version": "2.0.0" } }
```

### `GET /api/status`
Status atual do sistema de trading.
```json
{ "success": true, "data": {
  "running": false, "uptime": "0:00:00",
  "total_capital": 1200.0, "paper_trading": true
} }
```

### `POST /api/start`
Inicia o sistema em background.
- **200:** `{ "success": true, "data": { "message": "Trading system started successfully" } }`
- **400:** sistema já rodando · **500:** falha na inicialização

### `POST /api/stop`
Parada graciosa (fecha posições com `_shutdown`).
- **400:** sistema não está rodando

### `POST /api/emergency-stop`
Parada imediata (`_emergency_shutdown`, fecha tudo a mercado).

---

## Metrics

### `GET /api/metrics`
Métricas agregadas de trading.
```json
{ "success": true, "data": {
  "total_pnl": 0.0, "total_roi": 0.0, "daily_pnl": 0.0,
  "active_trades": 0, "win_rate": 0.0, "total_trades": 0,
  "max_drawdown": 0.0, "daily_trades": 0
} }
```

### `GET /api/performance-metrics`
Uso de recursos do processo (usa `psutil` se disponível).
```json
{ "success": true, "data": {
  "system_metrics": { "memory_percent": 45.2, "memory_available_mb": 512,
    "cpu_percent": 12.5, "gc_collections": 150, "gc_collected": 3200 },
  "application_metrics": { "logs_utilization": 23.0, "active_logs": 23, "max_logs": 1000 },
  "trading_metrics": { "system_running": true, "task_status": "running" },
  "timestamp": "2025-06-01T12:00:00"
} }
```

### `GET /api/equity`
Últimos 100 pontos da curva de equity. `timestamp` em **milissegundos**.
```json
{ "success": true, "data": { "equity_curve": [ { "timestamp": 1717243200000, "equity": 1200.0 } ] } }
```

### `GET /api/daily-stats`
P&L diário (últimos 30 dias, do banco ou memória).
```json
{ "success": true, "data": { "daily_stats": [ { "date": "2025-06-01", "pnl": 12.5, "trades": 4, "wins": 3, "losses": 1 } ] } }
```

---

## Bots

### `GET /api/bots`
Lista todos os bots. **Cada bot inclui `id`** (`symbol_timeframe`), usado em toggle/config.
```json
{ "success": true, "data": { "bots": [
  { "id": "LINK/USDT_5m", "symbol": "LINK/USDT", "timeframe": "5m",
    "status": "running", "pnl": 0.0, "trades": 0, "enabled": true }
] } }
```

### `GET /api/bots/count`
```json
{ "success": true, "data": { "current_count": 2, "maximum_allowed": 5, "can_add_more": true } }
```

### `GET /api/bots/available-symbols` / `GET /api/bots/available-timeframes`
```json
{ "success": true, "data": { "symbols": ["LINK/USDT", "BTC/USDT", "..."] } }
{ "success": true, "data": { "timeframes": ["1m", "5m", "15m", "..."] } }
```

### `GET /api/bots/{bot_id:path}/config`
Config completa de um bot. `bot_id` pode conter `/` (ex: `LINK/USDT_5m`).
- **503:** sistema não rodando · **404:** bot não encontrado

### `POST /api/bots/{bot_id:path}/toggle`
Body opcional `{ "enabled": true|false }`. Sem body, inverte o estado.
```json
{ "success": true, "data": { "bot_id": "LINK/USDT_5m", "enabled": false } }
```

### `POST /api/bots/add`
Body: `NewBotConfig` (symbol, timeframe, capital_allocation, max_risk_per_trade, +opcionais).
- **409:** bot já existe · **503:** sistema não rodando

### `PUT /api/bots/update/{bot_index}` / `DELETE /api/bots/remove/{bot_index}`
Atualiza/remove por índice 0-based na lista de bots.
- **404:** índice fora do range

---

## Config

### `GET /api/config` / `POST /api/config`
Config simples: `{ trading_mode, total_capital, daily_loss_limit, daily_profit_target }`.
POST persiste em `system_config.json`.

### `GET /api/config/full`
Config completa. **Cada bot em `bot_configs` inclui `id`.**
```json
{ "success": true, "data": {
  "global_config": { "total_capital": 1200.0, "paper_trading": true, "..." : "..." },
  "bot_configs": [ { "id": "LINK/USDT_5m", "symbol": "LINK/USDT", "timeframe": "5m", "..." : "..." } ]
} }
```

### `GET /api/config/backups`
Lista arquivos `system_config_backup_*.json`.

### `PUT /api/config/global`
Body: dict parcial de campos do `GlobalConfig`.

### `PUT /api/config/bot/{bot_id:path}`
Body: dict parcial de campos do `BotConfig`. Usa o id string.

### `POST /api/config/bot`
Alias de `POST /api/bots/add`.

### `DELETE /api/config/bot/{bot_id:path}`
Remove bot pelo id string (não índice).

### `POST /api/config/restore/{filename}`
Copia o backup sobre `system_config.json`.

---

## Logs

### `GET /api/logs`
Últimos 100 logs **estruturados** (parseados de `trading_system.log`).
```json
{ "success": true, "data": { "logs": [
  { "timestamp": "12:00:00", "level": "INFO", "message": "...", "source": "production_trading_system" }
] } }
```

### `GET /api/logs/statistics`
```json
{ "success": true, "data": { "total_entries": 500, "error_count": 3, "warning_count": 12, "info_count": 485 } }
```

### `GET /api/logs/categories`
```json
{ "success": true, "data": { "categories": ["system", "trading", "ml", "api", "risk", "persistence"] } }
```

---

## Analytics

### `GET /api/signal-quality`
Qualidade dos sinais e scores por camada.
```json
{ "success": true, "data": {
  "current_quality_score": 0.72, "signals_evaluated": 150,
  "signals_passed": 108, "signals_rejected": 42,
  "avg_quality_score": 0.68, "pass_rate": 0.72,
  "layer_scores": { "technical": 0.75, "market_structure": 0.65,
                    "binance_sentiment": 0.70, "ml_confidence": 0.80 },
  "recent_rejections": [ { "symbol": "LINK/USDT", "quality_score": 0.45, "timestamp": "12:00:00" } ]
} }
```

### `GET /api/market-regime/{symbol:path}`
Regime de mercado. `regime` ∈ `trending_bull | trending_bear | ranging | high_volatility | transitional`.
```json
{ "success": true, "data": {
  "regime": "ranging", "confidence": 0.75, "regime_duration": 42,
  "trend_strength": 0.3, "volatility_level": 0.5, "volume_profile": "medium",
  "breakout_frequency": 0.2,
  "strategy_config": { "strategy_type": "mean_reversion", "max_trades_per_day": 6,
                       "quality_threshold": 0.65, "risk_per_trade": 0.015 },
  "factors": ["Win rate 50% over last 20 trades"]
} }
```

### `GET /api/market-sentiment/{symbol:path}`
Sentimento. `sentiment_label` ∈ `bullish | bearish | neutral`.
```json
{ "success": true, "data": {
  "sentiment_label": "neutral", "sentiment_score": 0.5, "confidence": 0.65,
  "factors": ["Based on recent trade results"],
  "raw_data": {
    "funding_rate": 0.0001,
    "long_short_ratio": { "current_ratio": 1.0, "sentiment": "neutral" },
    "open_interest": { "trend": "stable", "change_pct": 0.0 },
    "order_book": { "sentiment": "neutral", "spread_pct": 0.02 }
  }
} }
```

### `POST /api/backtest`
Dispara `run_backtest.py` em background. Resultado escrito em `backtest_results.json`.

### `GET /api/backtest/results`
Retorna o conteúdo de `backtest_results.json`, ou mensagem se ainda não houver.

---

## Manutenção desta doc

Ao adicionar/alterar qualquer endpoint:
1. Adicionar `summary`, `tags` e `response_description` no decorador em `api_server.py`.
2. Atualizar o **Índice de endpoints** e a seção da categoria correspondente aqui.
3. Se um componente React novo consome o endpoint, atualizar `docs/FRONTEND_ARCHITECTURE.md`.
