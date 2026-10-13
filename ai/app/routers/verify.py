from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from app.services.verify_service import VerifyService

router = APIRouter(prefix="/verify", tags=["verify"])
svc = VerifyService()


@router.post("/")
async def verify_ekyc(
    front: UploadFile = File(...),
    back: Optional[UploadFile] = None,
    selfie: Optional[UploadFile] = None,
    liveness: Optional[List[UploadFile]] = File(default=None),
):
    try:
        front_b = await front.read()
        back_b = await back.read() if back else None
        selfie_b = await selfie.read() if selfie else None
        liveness_frames = [await f.read() for f in (liveness or [])]
        res = await run_in_threadpool(svc.verify, front_b, back_b, selfie_b, liveness_frames)
        return {"success": True, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
