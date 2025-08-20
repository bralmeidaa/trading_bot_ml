from fastapi import APIRouter, HTTPException
from fastapi import Form
from pydantic import BaseModel, Field
from typing import Optional

from ..bot.manager import manager

router = APIRouter(prefix="/bots", tags=["bots"])


class StartBotRequest(BaseModel):
    id: str = Field(..., description="Unique bot id, e.g., btcusdt_1m")
    symbol: str = Field(..., description="Symbol like BTC/USDT")
    timeframe: str = Field("1m")
    poll_seconds: int = Field(5, ge=1, le=60)


@router.get("")
async def list_bots():
    return manager.list()


@router.post("/start")
async def start_bot(payload: StartBotRequest):
    try:
        res = manager.start(
            bot_id=payload.id,
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            poll_seconds=payload.poll_seconds,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{bot_id}/stop")
async def stop_bot(bot_id: str):
    res = manager.stop(bot_id)
    if res["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Bot not found")
    return res


# Convenience: GET stop route for quick testing via browser links
@router.get("/{bot_id}/stop")
async def stop_bot_get(bot_id: str):
    res = manager.stop(bot_id)
    if res["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Bot not found")
    return res


# Form-friendly start endpoint (accepts form-data / x-www-form-urlencoded)
@router.post("/start-form")
async def start_bot_form(
    id: str = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form("1m"),
    poll_seconds: int = Form(5),
):
    try:
        res = manager.start(
            bot_id=id, symbol=symbol, timeframe=timeframe, poll_seconds=poll_seconds
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
