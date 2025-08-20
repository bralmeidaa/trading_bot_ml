from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .api.market import router as market_router
from .api.bot import router as bot_router
from .api.backtest import router as backtest_router
from .api.ui import router as ui_router
from .data.db import init_db
import logging

app = FastAPI(title="Trading Bot ML", version="0.1.0")

# Ensure our logs appear in the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# CORS for local development and future frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(market_router)
app.include_router(bot_router)
app.include_router(backtest_router)
app.include_router(ui_router)


@app.on_event("startup")
async def on_startup():
    # Create tables if they don't exist
    init_db()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": app.version}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Placeholder: later will push market data and model signals
            await websocket.send_text(f"echo: {data}")
    except WebSocketDisconnect:
        # Client disconnected
        pass
