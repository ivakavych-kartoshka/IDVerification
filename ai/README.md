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

## Document tamper detection (anti-tampering for CCCD)

`app/services/tamper_service.py` detects *digital* manipulation of the ID card
image itself - a swapped portrait or an inpainted text field (name, dates) -
as opposed to the QR cross-check (a data-consistency signal) or the liveness
checks (a *physical* presentation signal). It runs on the front image inside
`/verify/` and is also exposed standalone via `/tamper/analyze`.

### Two-tier engine

1. **Neural model (`TamperNet`)** - primary. An ultra-light multi-stream
   forensic network (~492K params) inspired by TruFor (CVPR'23) and the
   DeepID@ICCV 2025 entrants:
   - **RGB stream** - appearance / layout.
   - **Forensic stream** - a fixed (non-learned) 4-filter SRM 5x5 high-pass
     bank, to expose resampling and blending traces.
   - **Frequency stream** - block-DCT 8x8 AC/DC energy ratio (differentiable),
     to expose the compression mismatch a spliced region leaves behind.
   These fuse into two heads: a **classification head** (`logit`, bona fide vs
   forged) and a **localization head** (`mask_logits`, a per-pixel map of the
   suspected region, exposed as bounding boxes in the response).

2. **Classical forensic fallback** - four cv2/numpy signals used when the model
   is unavailable and *fused* with it when present:
   `boundary_energy` (splice seam), `noise_step` (noise-floor discontinuity),
   `blockiness` (JPEG grid mismatch), `low_hf_fraction` (lost high-frequency
   detail from inpainting).

The final score is a weighted fusion, `AI_TAMPER_MODEL_WEIGHT` on the model and
the remainder on the heuristic signals, renormalised when a signal is missing
(the card back has no portrait, for example).

Preprocessing: the image is letterboxed to 256x256 (divisible by 8 for the block
DCT) and normalised as RGB in **[0, 1]** - deliberately **not** 0-mean
standardised, so the DCT/SRM ratios stay meaningful.

### Safety rule (important)

The heuristic-only engine can **never** return `TAMPERED`; it caps at
`SUSPICIOUS`. A hard reject requires the neural model to have actually loaded.
Every result carries `model_loaded` so a silent fallback can never quietly
weaken the security decision. This is deliberate: an unvalidated heuristic must
not be able to reject a genuine citizen's card on its own.

| Variable | Default | Description |
| --- | --- | --- |
| `AI_TAMPER_ENABLED` | `true` | Run the tamper check inside `/verify/`. |
| `AI_TAMPER_MODEL` | `models/tamper_net.pth` | Checkpoint path; a `.onnx` file switches to the onnxruntime backend. |
| `AI_TAMPER_THRESHOLD` | `0.55` | Score at/above which the verdict is `TAMPERED`. |
| `AI_TAMPER_SUSPECT` | `0.35` | Score at/above which the verdict is `SUSPICIOUS`. |
| `AI_TAMPER_MODEL_WEIGHT` | `0.70` | Fusion weight on the neural score (rest goes to heuristics). |

`AI_TAMPER_THRESHOLD` / `AI_TAMPER_SUSPECT` / `SIGNAL_NORM` are **placeholders**
until fitted on real card data - see calibration below.

### Endpoints

- `POST /tamper/analyze` - multipart `file` -> `{status, tamper_score,
  model_prob, engine, model_loaded, signals, regions}`. Works on any image,
  independent of `/verify/`.
- `GET /tamper/healthz` - reports the active engine and whether the checkpoint
  loaded.

Inside `/verify/` the result appears as `tamper`, and a `TAMPERED` verdict adds
`tamper_reject` to the decision with a rejection reason.

### Preparing FantasyID

The Zenodo archive (CC BY 4.0) ships only `train` and `test` splits, laid out as
`<split>/bonafide/<device>/` and `<split>/attack/<method>/<device>/`, with a
per-image `.json` sidecar. The sidecar's `regions[]` carry
`region_attributes.region_provenance` = `original` / **`altered`**; the altered
boxes are the localization GT. Those boxes are in *pre-crop* coordinates, so
`prepare_fantasyid.py` applies each sidecar's
`cropping_info.transformation_matrix_into_cropped` homography before
rasterizing, then writes a `bona_fide/` + `forged/` + `masks/` layout:

```bash
python training/prepare_fantasyid.py ^
    --src D:/datasets/fantasyid/extract/FantasyID/train ^
    --csv D:/datasets/fantasyid/extract/FantasyID/train.csv ^
    --out D:/datasets/fantasyid/prepared/train
python training/prepare_fantasyid.py ^
    --src D:/datasets/fantasyid/extract/FantasyID/test ^
    --csv D:/datasets/fantasyid/extract/FantasyID/test.csv ^
    --out D:/datasets/fantasyid/prepared/test
```

The restricted `val` split is **not** in the public archive (it needs the
academic-license release at Zenodo `17063494`), so an internal split of `train`
is used for early stopping and the official `test` split for the benchmark F1.

### Training

```bash
cd ai
python training/train_tamper.py --data D:/datasets/fantasyid/prepared/train ^
    --epochs 40 --batch-size 16 --val-manifest models/fantasyid_val.txt ^
    --out models/tamper_net.pth
```

`CardTamperDataset` auto-detects the dataset layout: `bona_fide/` + `forged/`
folders, a flat directory with the label in the filename (`*_live.*` = bona
fide, `*_attack*` / `*_forged.*` = forged), or a split subdirectory (pass
`--data root/<split>`). Localization masks are picked up from `masks/` (also
`mask/`, `gt_masks/`, `localization/`) by matching stem; when no masks exist the
mask loss is auto-zeroed (`--mask-weight 0` / `--no-masks`) and only the
detection head trains. `--pos-weight` compensates class imbalance.

The reference benchmark is **FantasyID** (DeepID@ICCV 2025,
https://zenodo.org/records/17063366): 786 bona fide + 1572 forged train-val, 300
+ 1085 test. The reported metric is F1 @ 0.5 (DeepID Track 1); the TruFor
baseline shipped by the organizers is challenged by this dataset (FNR ~50% at
FPR 10%).

### Calibration

```bash
python training/calibrate_tamper.py --data D:/datasets/fantasyid/prepared/test ^
    --weights models/tamper_net.pth --out models/tamper_calibration.json
```

Sweeps the decision threshold on the **test** split to maximise image-level F1,
reports the FPR = 10% operating point (how DeepID reports, for a like-for-like
comparison with TruFor), and measures each heuristic signal's class separation
(|delta median| between bona fide and forged) so near-useless signals can have
their `SIGNAL_WEIGHTS` lowered. Recommended thresholds are written to JSON for
the record.

### Export to ONNX

```bash
python training/export_tamper_onnx.py --weights models/tamper_net.pth --out models/tamper.onnx
```

Exports opset 18 with two outputs, `logit` and `mask_logits`, and reports the
torch-vs-ONNX difference. Point `AI_TAMPER_MODEL` at the `.onnx` to serve it.

### Testing

```bash
python -m pytest tests/test_tamper.py -v
```

Covers the architecture and output shapes, letterbox geometry, the heuristic
signals, the contract that the heuristic-only engine never rejects on its own,
and the response schema.



