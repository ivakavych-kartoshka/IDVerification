from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from app.services.qr_service import QRService

router = APIRouter(prefix="/qr", tags=["qr"])
qr_service = QRService()


@router.post("/extract")
async def extract_qr(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = await run_in_threadpool(qr_service.decode, content, file.filename or "image.jpg")
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/healthz")
async def qr_health():
    return {"status": "ok"}
