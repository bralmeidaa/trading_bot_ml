# Trading Bot ML — Frontend Architecture (React)

Guia da aplicação React em `frontend_react/`. Leia junto com `docs/API_REFERENCE.md`.

---

## Stack

- **React 18** + **Vite** (build/dev server)
- **TailwindCSS** (estilo)
- **lucide-react** (ícones)
- **Chart.js** (gráfico de equity)
- Sem state manager externo — estado local + hooks customizados.

---

## Estrutura de diretórios

```
frontend_react/
├── index.html              # entry HTML; <script src="/src/main.jsx">
├── vite.config.js          # dev proxy /api → localhost:12000; build → dist/
├── dist/                   # build de produção (servido pelo FastAPI)
│   ├── index.html          # referencia /assets/*.js e /assets/*.css
│   └── assets/             # bundles JS/CSS (montados em /assets pelo backend)
└── src/
    ├── main.jsx            # ReactDOM.createRoot → <App/>
    ├── App.jsx             # navegação por abas (tabs) + layout
    ├── components/         # componentes de UI
    ├── hooks/useApi.js     # hook genérico de fetch + hooks por recurso
    ├── services/api.js     # cliente HTTP (apiService)
    └── utils/              # formatters.js, notifications.js
```

---

## Fluxo de dados

```
Componente
   │  usa
   ▼
hook (useMetrics, useBots, …)  ── ou ──  apiService.get(...) direto
   │  chama
   ▼
apiService.request()  →  fetch('/api/...')  →  FastAPI
   │  retorna JSON cru: { success, data }
   ▼
useApi: if (result.success) setData(result.data)
   │
   ▼
hook desembrulha o sub-objeto (data.bots, data.equity_curve, …)
```

### O contrato `{success, data}` (CRÍTICO)

O backend **sempre** responde `{ "success": true, "data": <payload> }`.
- `useApi.js` (linha ~23) checa `result.success` e guarda `result.data`.
- Componentes que usam `fetch()` cru (sem `apiService`) **devem** desembrulhar:
  ```js
  const json = await response.json();
  setData(json.data ?? json);   // defensivo
  ```
  Aplicável a: `SignalQualityMonitor`, `MarketRegimeMonitor`, `MarketSentimentPanel`.

Se o contrato quebrar (backend retorna flat sem `success`), **todos os dados aparecem
zerados/vazios** — foi exatamente o bug que originou esta refatoração.

---

## Navegação (App.jsx)

Navegação por abas via `activeView` (useState). Não usa React Router para tabs,
mas o backend tem catch-all SPA para deep links diretos.

| Aba | Componente |
|-----|-----------|
| `dashboard` | `PerformanceMetrics` + `EquityChart` + `BotStatusTable` + `TradesAndLogs` |
| `quality` | `SignalQualityMonitor` |
| `sentiment` | `MarketSentimentPanel` |
| `regime` | `MarketRegimeMonitor` |
| `config` | `ConfigurationPanel` |
| `logs` | `LogViewer` |

> `Header` e `Notifications` são sempre montados (fora das abas).
> `EnhancedConfigPanel`, `BotManagement`, `PerformanceMonitor`, `BotConfigModal` existem
> no código mas **não estão montados** em `App.jsx` atualmente (componentes disponíveis para uso futuro).

---

## Mapa componente → hook → endpoint → campos consumidos

| Componente | Hook / chamada | Endpoint | Campos acessados |
|-----------|----------------|----------|------------------|
| `Header` | `useSystemStatus` + `apiService.start/stop/emergencyStop` | `/status`, `/start`, `/stop`, `/emergency-stop` | `running`, `uptime`, `total_capital`, `paper_trading` |
| `PerformanceMetrics` | `useMetrics` | `/metrics` | `total_pnl`, `total_roi`, `daily_pnl`, `active_trades`, `win_rate`, `total_trades`, `max_drawdown`, `daily_trades` |
| `EquityChart` | `useEquity` | `/equity` | `equity_curve[].timestamp`, `equity_curve[].equity` |
| `BotStatusTable` | `useBots` + `apiService.toggleBot` | `/bots`, `/bots/{id}/toggle` | `bot.id`, `symbol`, `timeframe`, `status`, `pnl`, `trades`, `enabled` |
| `TradesAndLogs` | `useRecentTrades` + `useLogs` | `/trades/recent`, `/logs` | trade: `id, symbol, direction('long'/'short'), pnl, status, time, entry_price, exit_price` · log: `timestamp, level, message, source` |
| `ConfigurationPanel` | `useConfig` + `apiService.updateConfig` | `/config/full`, `/config` | `global_config.*`, `bot_configs[].*` |
| `SignalQualityMonitor` | `fetch` cru (5s) | `/signal-quality` | `current_quality_score`, `pass_rate`, `layer_scores.*`, `recent_rejections[]` |
| `MarketRegimeMonitor` | `fetch` cru (15s) | `/market-regime/{symbol}` | `regime`, `confidence`, `trend_strength`, `strategy_config.*`, `factors[]` |
| `MarketSentimentPanel` | `fetch` cru (10s) | `/market-sentiment/{symbol}` | `sentiment_label`, `sentiment_score`, `raw_data.*`, `factors[]` |
| `LogViewer` | `apiService.getLogs` + `/logs/statistics` + `/logs/categories` | `/logs`, `/logs/statistics`, `/logs/categories` | `logs[]`, statistics, categories |
| `PerformanceMonitor`* | `apiService.getPerformanceMetrics` (60s) | `/performance-metrics` | `system_metrics.*`, `application_metrics.*`, `trading_metrics.*` |

\* não montado em App.jsx atualmente.

---

## Hooks (`src/hooks/useApi.js`)

- **`useApi(apiCall, deps, interval)`** — base. Faz polling se `interval` setado, com
  **circuit breaker**: após 5 erros consecutivos, pausa 2 min.
- **`useSystemStatus()`** — `/status`, polling 15s
- **`useMetrics()`** — `/metrics`, polling 15s
- **`useEquity()`** — `/equity`, polling 30s, valida timestamps/equity
- **`useBots()`** — `/bots`, polling 30s, desembrulha `data.bots`
- **`useRecentTrades()`** — `/trades/recent`, polling 30s, aplica defaults seguros
- **`useLogs()`** — `/logs`, polling 20s
- **`useConfig()`** — `/config/full`, sem polling

### Loading sem flicker (importante)

`useApi` mantém `loading=true` **apenas até a primeira resposta**. Em refetches de
polling, os dados antigos continuam na tela enquanto a atualização acontece em background
— **não** volta para o skeleton. Isso elimina o "piscar" da UI a cada ciclo.

> Se adicionar um novo componente, **não** renderize skeleton baseado em `loading`
> após o primeiro load. Use `loading && !data` se precisar de um estado inicial.

### Intervalos de polling (resumo)

| Recurso | Intervalo |
|---------|-----------|
| status, metrics | 15s |
| logs | 20s |
| equity, bots, trades | 30s |
| signal-quality | 5s |
| market-sentiment | 10s |
| market-regime | 15s |
| performance-metrics | 60s |

---

## Cliente HTTP (`src/services/api.js`)

- `API_BASE_URL = '/api'` — **não** prefixe `/api` nos métodos (causa `/api/api/...`).
  ✅ `apiService.get('/config/full')`  ❌ `apiService.get('/api/config/full')`
- `request()` faz `fetch`, lança em `!response.ok` (lê `detail`), retorna o JSON cru.
- Métodos: `get/post/put/delete` + convenientes (`getMetrics`, `getBots`, `toggleBot`, etc).

---

## Guia: adicionar um novo endpoint + componente

1. **Backend** (`api_server.py`): criar o endpoint retornando `ok(payload)` (envelope `{success,data}`),
   com `tags=`, `summary=`, `response_description=`.
2. **Doc API**: adicionar linha no índice e seção em `docs/API_REFERENCE.md`.
3. **Cliente** (`services/api.js`): adicionar método (ex: `getFoo() { return this.get('/foo'); }`) — **sem** `/api`.
4. **Hook** (opcional, `hooks/useApi.js`): criar `useFoo()` desembrulhando o sub-objeto.
5. **Componente**: consumir via hook. Se usar `fetch` cru, desembrulhar com `json.data ?? json`.
6. **Doc Frontend**: adicionar linha no mapa componente→hook→endpoint acima.

---

## Build & Deploy

```bash
cd frontend_react
npm install
npm run build          # gera dist/ com index.html + assets/
```

O FastAPI serve:
- `/` → `dist/index.html`
- `/assets/*` → `dist/assets/*` (montado via StaticFiles)
- `/{qualquer-rota}` → `dist/index.html` (catch-all SPA)

Dev local: `npm run dev` (porta 44261, proxy `/api` → `localhost:12000`).
