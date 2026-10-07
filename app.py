#!/usr/bin/env python3
"""FastAPI wrapper cho pipeline xóa watermark/logo — deploy lên Hugging Face Spaces.

Kiến trúc: n8n Cloud Form (public) -> HTTP POST /clean -> ảnh sạch trả về trực tiếp.
- POST /clean (multipart field "image", tối đa 10MB): chạy pipeline attempts 0..2,
  trả về ảnh JPEG bytes, kèm headers X-QC-Score / X-QC-Pass / X-Method / X-Attempts.
- POST /clean?format=json: trả JSON {"image_base64": ..., "qc_score": ..., ...}.
- GET /: trang upload HTML đơn giản cho người dùng trực tiếp.
- GET /health: kiểm tra sống.
"""
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402

from lib import find_lama_model, log, qc_metrics, run_pipeline  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Xóa watermark/logo ảnh sản phẩm")

_LAMA = None


def get_lama():
    global _LAMA
    if _LAMA is None:
        _LAMA = find_lama_model()
    return _LAMA


def clean_image(img_bgr: np.ndarray) -> dict:
    lama_model = get_lama()
    best = None
    for attempt in range(3):
        res = run_pipeline(
            img_bgr, attempt=attempt, method="auto",
            lama_model=lama_model if attempt >= 2 else None,
        )
        qc = qc_metrics(res["result"], res["hard_mask"])
        methods = sorted(res["methods"])
        info = {
            "attempt": attempt,
            "method": "+".join(methods) if methods else "none",
            "mask_area_px": int(res["hard_mask"].sum()),
            "qc_pass": bool(qc["pass"]),
            "qc_score": float(qc["score"]),
            "qc_reasons": qc["reasons"],
            "stages_ms": {k: float(v) for k, v in res["stages_ms"].items()},
        }
        best = (res["result"], info)
        log(f"clean attempt {attempt}: score={qc['score']} pass={qc['pass']}")
        if qc["pass"]:
            break
    return {"cleaned_bgr": best[0], "info": best[1]}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/debug")
async def debug(request: Request):
    """Echo back what the request carried — for diagnosing client issues."""
    form = await request.form()
    fields = []
    for key, val in form.multi_items():
        if hasattr(val, "read"):
            data = await val.read()
            fields.append({
                "field": key,
                "filename": getattr(val, "filename", None),
                "content_type": getattr(val, "content_type", None),
                "bytes": len(data),
            })
        else:
            fields.append({"field": key, "value": str(val)[:200]})
    return {"fields": fields, "content_type": request.headers.get("content-type")}


@app.post("/clean")
async def clean(request: Request, format: str = "image"):
    # Lenient: accept the image from ANY multipart field (n8n/Zapier/etc.
    # may use different field names). Also accept raw body as fallback.
    form = await request.form()
    data: bytes | None = None
    field_name = ""
    for key, val in form.multi_items():
        if hasattr(val, "read"):
            data = await val.read()
            field_name = key
            break
    if data is None:
        body = await request.body()
        if body:
            data, field_name = body, "raw-body"
    if not data:
        raise HTTPException(400, "Không nhận được file ảnh nào (gửi multipart field bất kỳ).")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ảnh vượt quá 10MB.")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không đọc được ảnh (bytes không phải định dạng ảnh).")
    try:
        out = clean_image(img)
    except Exception as e:  # noqa: BLE001 - never 500 silently; log + report
        log(f"clean_image failed on field {field_name!r}: {e!r}")
        raise HTTPException(500, f"Lỗi xử lý ảnh: {type(e).__name__}")
    ok, buf = cv2.imencode(".jpg", out["cleaned_bgr"],
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "Lỗi encode ảnh kết quả.")
    info = out["info"]
    if format == "json":
        return {
            "image_base64": base64.b64encode(bytes(buf)).decode(),
            **info,
        }
    return Response(
        content=bytes(buf),
        media_type="image/jpeg",
        headers={
            "X-QC-Pass": str(info["qc_pass"]).lower(),
            "X-QC-Score": f"{info['qc_score']:.1f}",
            "X-Method": info["method"],
            "X-Attempts": str(info["attempt"] + 1),
        },
    )


INDEX_HTML = """<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xóa watermark/logo ảnh sản phẩm</title>
<style>body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
#drop{border:2px dashed #888;border-radius:12px;padding:40px;text-align:center;cursor:pointer}
img{max-width:100%;border-radius:8px;margin-top:12px}#meta{color:#555;font-size:14px}</style>
</head><body>
<h2>🧹 Xóa watermark / logo ảnh sản phẩm</h2>
<p>Thả ảnh vào khung dưới (tối đa 10MB) — hệ thống tự phát hiện và xóa, giữ nguyên phần còn lại.</p>
<div id="drop">Kéo thả ảnh vào đây hoặc bấm để chọn</div>
<input type="file" id="f" accept="image/*" hidden>
<div id="meta"></div><div id="out"></div>
<script>
const d=document.getElementById('drop'),f=document.getElementById('f');
d.onclick=()=>f.click();
['dragover','drop'].forEach(e=>d.addEventListener(e,x=>{x.preventDefault();}));
d.addEventListener('drop',x=>send(x.dataTransfer.files[0]));
f.onchange=()=>send(f.files[0]);
async function send(file){
 if(!file)return;
 document.getElementById('meta').textContent='Đang xử lý…';
 const fd=new FormData();fd.append('image',file);
 const r=await fetch('/clean',{method:'POST',body:fd});
 if(!r.ok){document.getElementById('meta').textContent='Lỗi: '+r.status;return;}
 const blob=await r.blob(),url=URL.createObjectURL(blob);
 document.getElementById('out').innerHTML=
  `<img src="${url}"><p><a href="${url}" download="cleaned.jpg">⬇ Tải ảnh đã xóa</a></p>`;
 document.getElementById('meta').textContent=
  `QC score: ${r.headers.get('X-QC-Score')} | pass: ${r.headers.get('X-QC-Pass')} | method: ${r.headers.get('X-Method')}`;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
