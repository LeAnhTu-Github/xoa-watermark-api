---
title: Xoa Watermark Anh San Pham
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Xóa watermark / logo ảnh sản phẩm

API + trang web đơn giản: upload ảnh → tự động phát hiện và xóa watermark/logo,
giữ nguyên bit-identical vùng không liên quan.

- `POST /clean` (multipart field `image`, ≤10MB) → ảnh JPEG đã xóa + headers `X-QC-Score`, `X-QC-Pass`, `X-Method`
- `POST /clean?format=json` → JSON `{image_base64, qc_score, ...}`
- `GET /` → trang upload cho người dùng trực tiếp
- `GET /health` → kiểm tra sống

Pipeline: detect (5 detectors OpenCV) → mask refine → inpaint (Telea / LaMa-ONNX nếu có model) → feathered composite → QC tự chấm điểm + retry tối đa 3 attempts.
