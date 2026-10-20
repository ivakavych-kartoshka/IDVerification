from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, ocr, qr, verify, liveness, tamper

app = FastAPI(title="AI eKYC Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(qr.router)
app.include_router(verify.router)
app.include_router(liveness.router)
app.include_router(tamper.router)


@app.get("/")
async def root():
    return {"message": "AI eKYC Service is running"}
