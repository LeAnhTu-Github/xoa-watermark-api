#!/usr/bin/env python3
"""FastAPI wrapper cho pipeline xóa watermark/logo — deploy trên Render (free).

Kiến trúc: n8n Cloud Form (public) -> HTTP POST /clean -> ảnh sạch trả về trực tiếp.
- POST /clean (multipart, field bất kỳ, tối đa 10MB): chạy pipeline attempts 0..2,
  trả về ảnh JPEG bytes, kèm headers X-QC-Score / X-QC-Pass / X-Method / X-Attempts.
- POST /clean?format=json: trả JSON {"image_base64": ..., "qc_score": ..., ...}.
- POST /batch (multipart: 1 file .zip HOẶC nhiều ảnh): xử lý async, trả về job_id;
  GET /batch/{job_id} xem tiến trình; GET /batch/{job_id}/result tải ZIP ảnh sạch.
- GET /: trang upload HTML đơn giản cho người dùng trực tiếp.
- GET /health: kiểm tra sống.
"""
import base64
import io
import json
import re
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402

from lib import find_lama_model, log, qc_metrics, run_pipeline  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 120
app = FastAPI(title="Xóa watermark/logo ảnh sản phẩm")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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


# ---------------------------------------------------------------- batch (/batch)
# Async batch cho 50-100 ảnh/lần: nhận 1 file .zip HOẶC nhiều ảnh trong 1 request,
# xử lý song song bằng thread pool, trả về job_id ngay; client poll tiến trình
# rồi tải 1 file ZIP chứa toàn bộ ảnh sạch (giữ nguyên tên file).

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _job_paths(job_id: str) -> dict:
    d = JOBS_DIR / job_id
    return {"dir": d, "status": d / "status.json", "zip": d / "cleaned.zip"}


def _write_status(job_id: str, **kw):
    p = _job_paths(job_id)["status"]
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            cur = {}
    cur.update(kw)
    p.write_text(json.dumps(cur))


def _process_one(args) -> tuple:
    """Xử lý 1 ảnh -> (tên file out, bytes jpg, qc_pass). Chạy trong worker thread."""
    name, data = args
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"không đọc được ảnh {name}")
    out = clean_image(img)
    ok, buf = cv2.imencode(".jpg", out["cleaned_bgr"],
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"không encode được {name}")
    stem = Path(name).stem
    return f"{stem}.jpg", bytes(buf), bool(out["info"]["qc_pass"])


def _run_batch_job(job_id: str, items: list):
    """Chạy nền: xử lý từng ảnh song song, gói ZIP, cập nhật tiến trình."""
    paths = _job_paths(job_id)
    total = len(items)
    done = 0
    passed = 0
    results = []
    try:
        for name, data, qc_ok in _BATCH_EXECUTOR.map(_process_one, items):
            results.append((name, data))
            done += 1
            passed += 1 if qc_ok else 0
            _write_status(job_id, status="processing", done=done, total=total)
        with zipfile.ZipFile(paths["zip"], "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in results:
                zf.writestr(name, data)
        _write_status(job_id, status="done", done=total, total=total,
                      qc_pass=passed, zip_bytes=paths["zip"].stat().st_size)
        log(f"batch {job_id}: xong {total} ảnh ({passed} QC pass)")
    except Exception as e:  # noqa: BLE001
        log(f"batch {job_id} lỗi: {e!r}")
        _write_status(job_id, status="failed", done=done, total=total,
                      error=f"{type(e).__name__}: {e}")


def _prune_old_jobs(max_age_h: int = 24):
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > max_age_h * 3600:
            for f in d.rglob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                d.rmdir()
            except OSError:
                pass


@app.post("/batch")  # 200 (not 202): n8n HTTP node drops 202 responses -> 0 items
async def batch_submit(request: Request, client_id: str = None):
    """Nhận 1 file .zip HOẶC nhiều ảnh (multipart field bất kỳ) -> {job_id,...}.
    client_id (optional): nếu truyền, dùng làm job_id luôn
    (để n8n không cần đọc job_id từ response)."""
    form = await request.form()
    items: list[tuple[str, bytes]] = []
    for key, val in form.multi_items():
        if not hasattr(val, "read"):
            continue
        data = await val.read()
        if not data:
            continue
        fname = getattr(val, "filename", None) or f"upload_{key}"
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        ext = Path(info.filename).suffix.lower()
                        if ext in _IMG_EXTS:
                            items.append((Path(info.filename).name,
                                          zf.read(info.filename)))
            except zipfile.BadZipFile:
                raise HTTPException(400, "File ZIP không hợp lệ.")
        elif Path(fname).suffix.lower() in _IMG_EXTS:
            items.append((fname, data))
    if not items:
        raise HTTPException(400, "Không nhận được ảnh nào (gửi 1 file ZIP hoặc nhiều ảnh).")
    if len(items) > MAX_BATCH_FILES:
        raise HTTPException(400, f"Tối đa {MAX_BATCH_FILES} ảnh/lần (nhận {len(items)}).")
    total_bytes = sum(len(b) for _, b in items)
    if total_bytes > 300 * 1024 * 1024:
        raise HTTPException(413, "Tổng dung lượng batch vượt quá 300MB.")

    job_id = uuid.uuid4().hex[:12]
    if client_id and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", client_id):
        job_id = client_id  # n8n gui len de tu biet job_id, khong can doc response
    paths = _job_paths(job_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _write_status(job_id, status="queued", done=0, total=len(items),
                  created=time.time())
    _prune_old_jobs()
    thread = threading.Thread(target=_run_batch_job, args=(job_id, items),
                              daemon=True)
    thread.start()
    _write_status(job_id, status="processing", done=0, total=len(items))
    base = str(request.base_url).rstrip("/")
    return {
        "job_id": job_id,
        "total": len(items),
        "status_url": f"{base}/batch/{job_id}",
        "result_url": f"{base}/batch/{job_id}/result",
    }


@app.get("/batch/{job_id}")
def batch_status(job_id: str):
    p = _job_paths(job_id)["status"]
    if not p.exists():
        raise HTTPException(404, "Job không tồn tại (có thể đã quá 24h hoặc service vừa restart).")
    return json.loads(p.read_text())


@app.get("/batch/{job_id}/result")
def batch_result(job_id: str):
    paths = _job_paths(job_id)
    if not paths["zip"].exists():
        st = {}
        if paths["status"].exists():
            st = json.loads(paths["status"].read_text())
        raise HTTPException(425 if st.get("status") == "processing" else 404,
                            "ZIP chưa sẵn sàng (job đang chạy hoặc đã lỗi).")
    return FileResponse(paths["zip"], media_type="application/zip",
                        filename=f"anh-da-xoa-watermark-{job_id}.zip")


@app.get("/batch/{job_id}/page", response_class=HTMLResponse)
def batch_page(job_id: str):
    """Trang trạng thái batch: tiến trình realtime + nút tải ZIP khi xong."""
    return f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đang xử lý batch {job_id}</title>
<style>
body{{font-family:system-ui;max-width:560px;margin:60px auto;padding:0 16px;text-align:center}}
.bar{{height:22px;background:#eee;border-radius:11px;overflow:hidden;margin:24px 0}}
.fill{{height:100%;width:0;background:#4caf50;transition:width .5s}}
#dl{{display:none;margin-top:24px;padding:14px 32px;background:#4caf50;color:#fff;
border-radius:8px;text-decoration:none;font-size:18px}}
#msg{{color:#555}}
</style></head><body>
<h2>🧹 Đang xóa watermark hàng loạt</h2>
<div class="bar"><div class="fill" id="fill"></div></div>
<p id="msg">Đang bắt đầu…</p>
<a id="dl" href="/batch/{job_id}/result">⬇ Tải ZIP ảnh đã xóa watermark</a>
<script>
const jid="{job_id}";
async function tick(){{
  try{{
    const r=await fetch("/batch/"+jid);const s=await r.json();
    const pct=s.total?Math.round(s.done/s.total*100):0;
    document.getElementById("fill").style.width=pct+"%";
    if(s.status==="done"){{
      document.getElementById("msg").textContent=
        `Xong! ${{s.done}} ảnh đã xử lý (${{s.qc_pass??s.done}} đạt QC).`;
      document.getElementById("dl").style.display="inline-block";
      return;
    }}
    if(s.status==="failed"){{
      document.getElementById("msg").textContent="Lỗi: "+(s.error||"không rõ");
      return;
    }}
    document.getElementById("msg").textContent=`Đang xử lý ${{s.done}}/${{s.total}} ảnh…`;
  }}catch(e){{
    document.getElementById("msg").textContent="Mất kết nối, đang thử lại…";
  }}
  setTimeout(tick,4000);
}}
tick();
</script></body></html>"""


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


BATCH_UPLOAD_HTML = """<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Xóa watermark hàng loạt</title>
<style>body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
#drop{border:2px dashed #888;border-radius:12px;padding:40px;text-align:center;cursor:pointer}
img{max-width:100%;border-radius:8px;margin-top:12px}#meta{color:#555;font-size:14px}</style>
</head><body>
<h2>📦 Xóa watermark hàng loạt</h2>
<p>Upload 1 file <b>.zip</b> chứa tối đa 100 ảnh sản phẩm (tổng &lt; 300MB). Hệ thống xử lý từng ảnh rồi gói thành 1 file ZIP để tải về.</p>
<div id="drop">Thả file .zip vào đây hoặc bấm để chọn file</div>
<input type="file" id="f" accept=".zip" hidden>
<div id="meta"></div><div id="out"></div>
<script>
const d=document.getElementById('drop'),f=document.getElementById('f');
d.onclick=()=>f.click();
['dragover','drop'].forEach(e=>d.addEventListener(e,x=>{x.preventDefault();}));
d.addEventListener('drop',x=>send(x.dataTransfer.files[0]));
f.onchange=()=>send(f.files[0]);
async function send(file){
if(!file)return;
document.getElementById('meta').textContent='Đang upload '+file.name+' ('+Math.round(file.size/1024)+' KB)...';
const fd=new FormData();fd.append('file',file);
const r=await fetch('/batch',{method:'POST',body:fd});
if(!r.ok){document.getElementById('meta').textContent='Lỗi: '+r.status;return;}
const j=await r.json();
window.location='/batch/'+j.job_id+'/page';
}
</script></body></html>"""


@app.get("/batch-upload", response_class=HTMLResponse)
def batch_upload():
    return BATCH_UPLOAD_HTML
