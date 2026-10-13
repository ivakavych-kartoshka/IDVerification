from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.active_liveness_service import ActiveLivenessService

router = APIRouter(prefix="/liveness", tags=["liveness"])
svc = ActiveLivenessService()


@router.get("/healthz")
async def liveness_health():
    return {"status": "ok"}


@router.post("/pose")
async def liveness_pose(file: UploadFile = File(...)):
    """Estimate head pose for a single frame (used for real-time guidance)."""
    try:
        content = await file.read()
        result = await run_in_threadpool(svc.pose, content)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/active")
async def liveness_active(frames: List[UploadFile] = File(...)):
    """Verify an ordered challenge sequence (straight + left + right)."""
    try:
        payload = [await f.read() for f in frames]
        result = await run_in_threadpool(svc.analyze, payload)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
