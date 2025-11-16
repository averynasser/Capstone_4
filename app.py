#!/usr/bin/env python3
"""
Streamlit app: Vehicle Detector + GPT-4o-mini chatbot
- Lazy-import ultralytics / cv2 to avoid startup import errors on cloud
- Auto-download model from MODEL_URL (set in Streamlit Secrets as MODEL_URL)
- Supports OpenAI new SDK (OpenAI client) and fallback handling
"""
import os
import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import requests

st.set_page_config(layout="wide", page_title="Vehicle Detector + GPT-4o-mini")

# ---------- Sidebar / config ----------
st.sidebar.header("Configuration")
dataset_path = st.sidebar.text_input("Dataset path (optional)", "/content/dataset")
model_path = st.sidebar.text_input("Model path", "models/model_best.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
show_boxes = st.sidebar.checkbox("Show bounding boxes", True)
api_key_input = st.sidebar.text_input("OpenAI API Key (optional)", os.getenv("OPENAI_API_KEY", ""), type="password")
use_small_model = st.sidebar.checkbox("Use small demo model (yolov8n)", False)

# prefer sidebar key over env
OPENAI_API_KEY = (api_key_input.strip() if api_key_input else os.getenv("OPENAI_API_KEY", "")).strip()
MODEL_URL = os.getenv("MODEL_URL", "").strip()  # recommended to set in Streamlit Secrets
os.makedirs(Path(model_path).parent, exist_ok=True)

st.title("🚗 Vehicle Detector + GPT-4o-mini Chatbot (Capstone 4)")

# ---------- Utilities ----------
def download_model(url: str, dest: str):
    """Download model from URL to dest (streaming)."""
    if not url:
        raise ValueError("MODEL_URL kosong.")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = resp.headers.get("content-length")
    tmp = dest + ".download"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)

def draw_boxes_cv2(img_pil: Image.Image, detections: list, names: dict):
    """Draw boxes using cv2 (if available). Returns PIL image."""
    try:
        import cv2
    except Exception:
        # fallback: return original image
        return img_pil
    img = np.array(img_pil.convert("RGB"))
    for d in detections:
        bbox = d.get("bbox", [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cls = int(d.get("class_id", -1))
            conf = float(d.get("conf", 0.0))
            label = f"{names.get(cls, str(cls))} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return Image.fromarray(img)

# ---------- Model loading (lazy) ----------
@st.cache_resource
def load_model(path: str, small_demo: bool = False):
    """Lazy import and load YOLO model. Returns (model, names_dict)."""
    try:
        # lazy import
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"ultralytics import gagal: {e}")

    # support demo small model (bundled from ultralytics hub) if requested
    if small_demo:
        model = YOLO("yolov8n.pt")
    else:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {path}")
        model = YOLO(str(path))

    # try extract class names if available
    names = {}
    try:
        names = getattr(model, "names", {}) or {}
    except Exception:
        names = {}
    return model, names

# ---------- UI: Input and detection ----------
col1, col2 = st.columns([1, 1])
with col1:
    st.header("Input")
    uploaded = st.file_uploader("Upload image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    imgs = []
    if uploaded:
        for u in uploaded:
            try:
                imgs.append((u.name, Image.open(u).convert("RGB")))
            except Exception as e:
                st.error(f"Gagal membuka file {u.name}: {e}")

with col2:
    st.header("Detection Results")
    if not imgs:
        st.info("Unggah gambar untuk menjalankan deteksi.")
    else:
        # ensure model exists / download if MODEL_URL provided
        if use_small_model:
            model_file = None
        else:
            model_file = model_path

        if not use_small_model and not Path(model_file).exists():
            if MODEL_URL:
                with st.spinner("Model tidak ditemukan – mengunduh dari MODEL_URL ..."):
                    try:
                        download_model(MODEL_URL, model_file)
                        st.success("Model selesai diunduh.")
                    except Exception as e:
                        st.error(f"Gagal mengunduh model dari MODEL_URL: {e}")
                        st.stop()
            else:
                st.error("Model tidak ada dan MODEL_URL tidak di-set (set MODEL_URL di Secrets atau gunakan model path valid).")
                st.stop()

        # attempt to load model
        try:
            model, names = load_model(model_file if model_file else "", small_demo=use_small_model)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            st.stop()

        # run detection for each image
        for fname, img in imgs:
            st.subheader(fname)
            try:
                res = model.predict(source=np.array(img), conf=conf_thresh, imgsz=640, verbose=False)
            except Exception as e:
                st.error(f"Gagal menjalankan prediksi pada {fname}: {e}")
                continue

            r = res[0]
            detections = []
            if hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    try:
                        xy = b.xyxy.cpu().numpy().flatten().tolist()
                    except Exception:
                        try:
                            xy = b.xyxy.tolist()
                        except Exception:
                            xy = []
                    try:
                        conf = float(b.conf.cpu().numpy())
                    except Exception:
                        conf = float(getattr(b, "conf", 0.0))
                    try:
                        cls = int(b.cls.cpu().numpy())
                    except Exception:
                        cls = int(getattr(b, "cls", -1))
                    if xy and conf >= conf_thresh:
                        detections.append({"bbox": xy, "conf": conf, "class_id": cls})

            counts = {}
            for d in detections:
                cname = names.get(d["class_id"], str(d["class_id"]))
                counts[cname] = counts.get(cname, 0) + 1

            if show_boxes:
                vis = draw_boxes_cv2(img, detections, names)
                st.image(vis, use_column_width=True)
            st.write("Counts:", counts)
            st.session_state["last_detections"] = {"file": fname, "detections": detections, "counts": counts}

# ---------- Chatbot panel ----------
st.markdown("---")
st.header("Chat with Detector (GPT-4o-mini)")
if "last_detections" not in st.session_state:
    st.info("Jalankan deteksi dulu sebelum chat.")
else:
    user_q = st.text_input("Tanyakan sesuatu terkait hasil deteksi (contoh: \"Ada berapa mobil?\")", key="chat_q")
    if st.button("Tanya GPT-4o-mini"):
        if not OPENAI_API_KEY:
            st.error("OPENAI API key kosong. Masukkan di sidebar atau set OPENAI_API_KEY sebagai Secret.")
        else:
            det_info = st.session_state["last_detections"]
            prompt = (
                "Kamu adalah asisten AI yang membantu menjelaskan hasil deteksi kendaraan.\n"
                f"Berikut data deteksi terakhir:\n{json.dumps(det_info, indent=2)}\n"
                f"Pertanyaan pengguna: {user_q}\n"
                "Jawablah singkat dan informatif dalam Bahasa Indonesia."
            )
            with st.spinner("Menghubungi OpenAI..."):
                try:
                    # use new OpenAI client (openai>=1.0)
                    from openai import OpenAI
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    # try chat.completions (if supported)
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role":"system","content":"Kamu adalah asisten deteksi kendaraan yang ramah dan informatif."},
                                {"role":"user","content":prompt},
                            ],
                            temperature=0.6,
                            max_tokens=400,
                        )
                        # extract
                        answer = None
                        if hasattr(resp, "choices") and len(resp.choices) > 0:
                            choice = resp.choices[0]
                            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                            if msg:
                                answer = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                            else:
                                answer = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
                    except Exception:
                        # fallback to unified Responses API
                        resp2 = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=400, temperature=0.6)
                        answer = getattr(resp2, "output_text", None)
                        if not answer:
                            # try parse structured output
                            outputs = getattr(resp2, "output", None) or (resp2.get("output") if isinstance(resp2, dict) else None)
                            if outputs:
                                parts = []
                                if isinstance(outputs, list):
                                    for p in outputs:
                                        if isinstance(p, dict):
                                            for c in p.get("content", []):
                                                text = c.get("text") if isinstance(c, dict) else None
                                                if text:
                                                    parts.append(text)
                                elif isinstance(outputs, dict):
                                    for c in outputs.get("content", []):
                                        text = c.get("text") if isinstance(c, dict) else None
                                        if text:
                                            parts.append(text)
                                if parts:
                                    answer = "\n".join(parts)

                    if not answer:
                        st.error("OpenAI tidak mengembalikan jawaban (cek model/akses/API key).")
                    else:
                        st.success(answer)
                except Exception as e:
                    st.error(f"Error saat memanggil OpenAI: {e}")
