from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from app.services.ocr_service import OCRService

router = APIRouter(prefix="/ocr", tags=["ocr"])
ocr_service = OCRService()


@router.post("/extract")
async def extract_ocr(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = await run_in_threadpool(ocr_service.extract, content, file.filename or "image.jpg")
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/healthz")
async def ocr_health():
    return {"status": "ok"}
