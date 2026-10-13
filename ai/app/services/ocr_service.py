from typing import Any, Dict, Optional
import io
import numpy as np
from PIL import Image, ImageOps
import re
import os


class OCRService:
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        # Try RapidOCR first (more stable on this env)
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._mode = "rapid"
            return self._ocr
        except Exception:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang="vi",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    use_angle_cls=False,
                )
                self._mode = "paddle"
                return self._ocr
            except Exception as e:
                raise RuntimeError(f"Cannot init OCR: {e}")

    def _to_array(self, content: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img)

    def extract(self, content: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
        img = self._to_array(content)
        ocr = self._get_ocr()
        mode = getattr(self, "_mode", "rapid")
        res = None
        try:
            if mode == "rapid":
                res = ocr(img)
            else:
                try:
                    res = ocr.predict(img)
                except Exception:
                    res = ocr.predict(filename)
        except Exception:
            res = None

        lines = []
        if isinstance(res, tuple):
            # rapidocr returns (results, elapse) or (det, rec, cls, ...)
            for part in res:
                if isinstance(part, list):
                    for item in part:
                        if len(item) >= 3:
                            lines.append({"text": str(item[1]), "score": float(item[2])})
                    break  # take first list
        elif isinstance(res, list) and len(res) > 0:
            r = res[0]
            if isinstance(r, dict):
                rec_texts = r.get("rec_texts") or r.get("texts") or []
                rec_scores = r.get("rec_scores") or r.get("scores") or []
                for t, s in zip(rec_texts, rec_scores):
                    lines.append({"text": t, "score": float(s) if isinstance(s, (int, float)) else 0.0})
            else:
                for item in r:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        box, textinfo = item[0], item[1]
                        if isinstance(textinfo, dict):
                            lines.append({"text": textinfo.get("text", ""), "score": float(textinfo.get("score", 0))})
                        elif len(item) >= 3 and isinstance(item[2], (int, float)):
                            lines.append({"text": str(textinfo), "score": float(item[2])})
                        else:
                            lines.append({"text": str(textinfo), "score": 0.0})

        text_all = " ".join(l["text"] for l in lines)
        doc_number = self._find_document_number(text_all)
        return {"lines": lines, "full_text": text_all, "document_number": doc_number}

    @staticmethod
    def _find_document_number(text: str) -> Optional[str]:
        """Pick the most plausible CCCD/CMND number from OCR text.

        CCCD numbers are 12 digits, legacy CMND numbers are 9. Prefer the longer,
        more specific match so dates (8 digits) or other tokens are not chosen.
        """
        if not text:
            return None
        for pattern in (r"\b\d{12}\b", r"\b\d{9}\b", r"\b\d{9,12}\b"):
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
