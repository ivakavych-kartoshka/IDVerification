from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from app.services.face_service import FaceService
from app.services.tamper_service import TamperService

router = APIRouter(prefix="/tamper", tags=["tamper"])
tamper_service = TamperService()


@router.post("/analyze")
async def analyze_tamper(front: UploadFile = File(...)):
    """Phân tích ảnh mặt trước thẻ: vùng giả mạo (thay ảnh/nội dung chữ)."""
    try:
        content = await front.read()
        img_rgb = FaceService.load_image(content)
        result = await run_in_threadpool(tamper_service.analyze, img_rgb)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/healthz")
async def tamper_health():
    return {"status": "ok", "enabled": tamper_service.enabled, "model_loaded": tamper_service.available()}
