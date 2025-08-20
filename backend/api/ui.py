from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from ..bot.manager import manager

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def ui_home():
    bots = manager.list()
    bots_rows = "".join(
        f"<tr><td>{b['id']}</td><td>{b['symbol']}</td><td>{b['timeframe']}</td><td>{b['running']}</td>"
        f"<td><a href='/bots/{b['id']}/stop'>Stop</a></td></tr>" for b in bots
    )
    html = f"""
    <html>
    <head>
        <title>Trading Bot UI</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
            th {{ background: #f2f2f2; }}
            form {{ margin: 16px 0; }}
            input[type=text], input[type=number] {{ padding: 6px; margin-right: 8px; }}
            button {{ padding: 6px 12px; }}
        </style>
    </head>
    <body>
        <h1>Trading Bot Control</h1>
        <h2>Start Bot (Form)</h2>
        <form method="post" action="/bots/start-form">
            <input type="text" name="id" placeholder="bot_btc_1m" required />
            <input type="text" name="symbol" placeholder="BTC/USDT" required />
            <input type="text" name="timeframe" value="1m" />
            <input type="number" name="poll_seconds" value="5" min="1" max="60" />
            <button type="submit">Start</button>
        </form>

        <h2>Backfill Market Data</h2>
        <form method="post" action="/market/backfill">
            <input type="text" name="symbol" placeholder="BTC/USDT" required />
            <input type="text" name="timeframe" value="1m" />
            <input type="number" name="limit" value="500" min="1" max="1000" />
            <button type="submit">Backfill</button>
        </form>

        <h2>Running Bots</h2>
        <table>
            <thead><tr><th>ID</th><th>Symbol</th><th>Timeframe</th><th>Running</th><th>Action</th></tr></thead>
            <tbody>
                {bots_rows}
            </tbody>
        </table>

        <p>API Docs: <a href="/docs">/docs</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
