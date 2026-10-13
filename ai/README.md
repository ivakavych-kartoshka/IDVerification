# AI eKYC Service

Simple FastAPI service for ID card OCR, document quality, face matching, and liveness.

## Structure
- `app/main.py` - FastAPI app
- `app/routers/` - HTTP endpoints
- `app/services/` - OCR, face, liveness, verification logic

## Run locally
```bash
cd ai
python -m venv .venv  # optional
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

## Endpoints
- GET `/healthz` - health check
- POST `/ocr/extract` - upload image, return OCR lines + full text + document_number
- POST `/qr/extract` - upload image, decode the CCCD QR code (raw payload + parsed fields)
- POST `/verify/` - multipart: `front`, `back?`, `selfie?` -> returns verification result

## Notes
- Uses RapidOCR (fallback PaddleOCR). 
- Face detection/recognition uses UltraFace (`models/version-RFB-640.onnx`, fixed 480x640 input) + MediaPipe + ArcFace (`models/recognition.onnx`).
- Passive liveness uses the MiniFASNetV2 silent anti-spoofing model (fallback: Laplacian variance).
- All images are EXIF-rotated on load, so photos straight from a phone are handled.
- The selfie must contain exactly one face (multiple faces are rejected).
- Heavy inference runs in a threadpool so endpoints stay responsive.

## Passive liveness (anti-spoofing)
`app/services/fasnet.py` runs the bundled PAD checkpoint
`models/2.7_80x80_MiniFASNetV2.pth` (MiniFASNetV2, 3 classes; index 1 = real,
0/2 = spoof print/replay) on an 80x80 face crop expanded x2.7 around the
detected face box. This is a **texture** check: it distinguishes a live face from
a printed photo or a screen replay, but **not** a 3D mask or a mannequin, and it
cannot detect a replayed *video*.

Preprocessing must match the reference project exactly: the crop is fed as BGR
in **[0, 255]** (the reference `to_tensor` deliberately skips the usual `/255`
step). Scaling to [0, 1] makes every genuine face score as spoof.

Tunable via environment variables (on the AI service):

| Variable | Default | Description |
| --- | --- | --- |
| `AI_FAS_ENABLED` | `true` | Enable the MiniFASNetV2 engine. |
| `AI_FAS_MODEL` | `models/2.7_80x80_MiniFASNetV2.pth` | Checkpoint path. |
| `AI_FAS_SCALE` | `2.7` | Face-box expansion factor for the crop. |
| `AI_FAS_REAL_THRESHOLD` | `0.5` | Min real-class probability to pass. |
| `AI_FAS_DEVICE` | `auto` | `auto`, `cpu` or `cuda`. |
| `AI_FAS_TTA` | `true` | Average the verdict over a horizontal flip (test-time augmentation). |

When torch or the checkpoint is unavailable, or if inference raises, the service
falls back to the Laplacian-variance blur check (threshold `25.0`).

## Active liveness (head-pose challenge)
`app/services/active_liveness_service.py` checks the straight / turn-left /
turn-right challenge from the landmarker's head pose (yaw). On top of the pose
checks it now also:

- rejects a set of **near-identical frames** (a static photo or a frozen screen
  replay) — `AI_ACTIVE_STATIC_CHECK` / `AI_ACTIVE_STATIC_DIFF`;
- reports a **blink** signal from face blendshapes (`AI_ACTIVE_REQUIRE_BLINK` to
  make a blink mandatory).

| Variable | Default | Description |
| --- | --- | --- |
| `AI_ACTIVE_STATIC_CHECK` | `true` | Reject when the frames are essentially static. |
| `AI_ACTIVE_STATIC_DIFF` | `1.5` | Mean grayscale diff below which frames count as static. |
| `AI_ACTIVE_REQUIRE_BLINK` | `false` | Require a detected blink to pass. |

## QR code cross-check
The QR on the back of a CCCD/CC card is decoded with OpenCV (falling back to
`pyzbar` if installed) and its structured fields (document number, name, DOB,
gender, ...) are cross-checked against the OCR text. This is a tampering signal,
not cryptographic proof (the printed QR can be copied). Result is returned as
`qr` and `qr_consistency` in `/verify/`.

Tunable via environment variables (on the AI service):

| Variable | Default | Description |
| --- | --- | --- |
| `AI_QR_ENABLED` | `true` | Enable QR decode + comparison. |
| `AI_QR_REQUIRE` | `false` | Reject when no readable QR is found. |
| `AI_QR_STRICT` | `true` | Reject when the QR document number is absent from the OCR text. |
| `AI_QR_STRICT_NAME` | `false` | Also reject when the QR full name does not match the OCR text. |

Decoded QR data is persisted by the backend in the `VerificationQr` table
(one row per verification) so it can be audited independently of the raw AI JSON.

## Testing the QR feature

### 1. Unit tests (no running services needed)

```bash
cd ai
python -m pytest tests/test_qr_service.py -v
```

Covers payload parsing (with/without the legacy CMND field), OCR cross-check
verdicts and decoding of the committed fixture `tests/fixtures/cccd_back_qr.png`.

### 2. Decode endpoint

Start the service, then post an image of the card back:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8001

curl -X POST http://127.0.0.1:8001/qr/extract ^
  -F "file=@tests/fixtures/cccd_back_qr.png;type=image/png"
```

The response contains `found`, the raw `payload`, parsed `fields` and `reader`.

### 3. Generate your own test image

```bash
pip install "qrcode[pil]"
python tests/make_qr_fixture.py "068096013455|251043755|Trần Chí Hào|08101996|Nam|Hà Nội|30082021" back_qr.png
```

### 4. End-to-end (UI)

Upload a real front image plus a back image that has a readable QR. On success
the verification detail page shows a **Thông tin QR** section and the QR row
(`QR ↔ OCR`) on the result screen.

## Testing passive liveness

```bash
cd ai
python -m pytest tests/test_liveness.py -v
```

Covers the crop geometry and a regression guard asserting the network input
stays in the [0, 255] range (a `/255` normalisation would flag every real face
as spoof). Tests skip automatically if torch or the checkpoint is missing.



