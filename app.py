
"""
app.py - Streamlit app with YOLO detection and GPT-4o-mini chatbot integration.
Requirements: ultralytics, streamlit, openai, pillow, opencv-python
Run:
    streamlit run app.py
Set OPENAI_API_KEY in sidebar or environment before using chatbot.
"""
import os
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import yaml
import json

# OpenAI import with fallback
try:
    import openai
except Exception:
    openai = None

st.set_page_config(layout="wide", page_title="Vehicle Detector + GPT-4o-mini")
st.title("🚗 Vehicle Detector + GPT-4o-mini Chatbot (Capstone 4)")

st.sidebar.header("Configuration")
dataset_path = st.sidebar.text_input("Dataset path", "vehicle-detection.v1i.yolov12")
model_path = st.sidebar.text_input("Model path", "models/model_best.pt")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
show_boxes = st.sidebar.checkbox("Show bounding boxes", True)
api_key = st.sidebar.text_input("OpenAI API Key (optional for chatbot)", os.getenv("OPENAI_API_KEY", ""), type="password")

if api_key and openai:
    openai.api_key = api_key

# load class names from data.yaml if present
names = {}
try:
    data_yaml_path = Path(dataset_path) / "data.yaml"
    if data_yaml_path.exists():
        with open(data_yaml_path, "r") as f:
            d = yaml.safe_load(f)
            names = d.get("names", {}) or {}
except Exception:
    names = {}

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

# check model path
if not Path(model_path).exists():
    st.sidebar.error("Model path tidak ditemukan. Pastikan model_best.pt ada di folder models/")
    st.stop()

# load model
try:
    model = load_model(model_path)
except Exception as e:
    st.sidebar.error(f"Gagal memuat model: {e}")
    st.stop()

# utilities
def draw_boxes_pil(img_pil: Image.Image, detections: list):
    """Draw detections (list of dicts with bbox [x1,y1,x2,y2], conf, class_id) on PIL image."""
    img = np.array(img_pil.convert("RGB"))
    for d in detections:
        bbox = d.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cls = int(d.get("class_id", -1))
            conf = float(d.get("conf", 0.0))
            label = f"{names.get(cls, str(cls))} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(img)

# UI layout
col1, col2 = st.columns([1, 1])
with col1:
    st.header("Input")
    uploaded = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
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
        for fname, img in imgs:
            st.subheader(fname)
            # predict
            try:
                res = model.predict(source=np.array(img), conf=conf_thresh, imgsz=640, verbose=False)
            except Exception as e:
                st.error(f"Gagal menjalankan prediksi pada {fname}: {e}")
                continue

            r = res[0]
            detections = []
            if hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    # robust extraction of bbox/score/class
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
                vis = draw_boxes_pil(img, detections)
                st.image(vis, use_column_width=True)
            st.write("Counts:", counts)
            st.session_state["last_detections"] = {"file": fname, "detections": detections, "counts": counts}

# Chatbot panel
st.markdown("---")
st.header("Chat with Detector (GPT-4o-mini)")

if "last_detections" not in st.session_state:
    st.info("Jalankan deteksi dulu sebelum chat.")
else:
    user_q = st.text_input("Tanyakan sesuatu terkait hasil deteksi (contoh: \"Ada berapa mobil?\")", key="chat_q")
    if st.button("Tanya GPT-4o-mini"):
        if not openai:
            st.error("Library OpenAI tidak tersedia. Install paket openai jika ingin gunakan chatbot.")
        elif not (api_key or os.getenv("OPENAI_API_KEY")):
            st.error("Masukkan API Key OpenAI di sidebar atau set OPENAI_API_KEY di environment.")
        else:
            det_info = st.session_state["last_detections"]
            prompt = (
                "Kamu adalah asisten AI yang membantu menjelaskan hasil deteksi kendaraan.\n"
                f"Berikut data deteksi terakhir:\n{json.dumps(det_info, indent=2)}\n"
                f"Pertanyaan pengguna: {user_q}\n"
                "Jawablah dengan bahasa alami, ringkas, dan informatif dalam Bahasa Indonesia."
            )
            with st.spinner("GPT-4o-mini sedang memproses..."):
                try:
                    # Pastikan openai module ada
                    if openai is None:
                        st.error("Library OpenAI tidak tersedia. Jalankan: pip install openai")
                    else:
                        # ambil API key dari sidebar (lebih prioritas) atau env
                        key = api_key.strip() if api_key else os.getenv("OPENAI_API_KEY", "").strip()
                        if not key:
                            st.error("API key kosong. Masukkan di sidebar atau set OPENAI_API_KEY di environment.")
                        else:
                            answer = None

                            # tentukan apakah memakai SDK baru (>=1.0)
                            ver = getattr(openai, "__version__", None)
                            use_new_sdk = False
                            if ver:
                                try:
                                    from packaging.version import parse as _parse
                                    use_new_sdk = _parse(ver) >= _parse("1.0.0")
                                except Exception:
                                    # fallback sederhana
                                    try:
                                        use_new_sdk = int(str(ver).split(".")[0]) >= 1
                                    except Exception:
                                        use_new_sdk = True  # assume new if unknown

                            # If new SDK available, use OpenAI client
                            if use_new_sdk:
                                try:
                                    from openai import OpenAI
                                    client = OpenAI(api_key=key)
                                except Exception as e:
                                    raise RuntimeError(f"Gagal inisialisasi OpenAI client: {e}")

                                # Try chat.completions first (most direct)
                                try:
                                    resp = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "Kamu adalah asisten deteksi kendaraan yang ramah dan informatif."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.6,
                                        max_tokens=400,
                                    )
                                    # extract answer robustly
                                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                                        choice = resp.choices[0]
                                        msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                                        if msg:
                                            answer = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                                        else:
                                            answer = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
                                except Exception:
                                    # fallback to Responses API if chat.completions unsupported
                                    try:
                                        resp2 = client.responses.create(
                                            model="gpt-4o-mini",
                                            input=prompt,
                                            max_tokens=400,
                                            temperature=0.6,
                                        )
                                        if hasattr(resp2, "output_text") and resp2.output_text:
                                            answer = resp2.output_text
                                        else:
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
                                    except Exception as e_resp:
                                        # bubble up for debugging
                                        raise RuntimeError(f"Responses API failed: {e_resp}")

                            else:
                                # Legacy path (should not be used for openai>=1.0, but kept as fallback)
                                try:
                                    resp_legacy = openai.ChatCompletion.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "Kamu adalah asisten deteksi kendaraan yang ramah dan informatif."},
                                            {"role": "user", "content": prompt},
                                        ],
                                        temperature=0.6,
                                        max_tokens=400,
                                        api_key=key if hasattr(openai, 'api_key') else None
                                    )
                                    answer = resp_legacy.get("choices", [{}])[0].get("message", {}).get("content")
                                except Exception as e_leg:
                                    raise RuntimeError(f"Legacy ChatCompletion failed: {e_leg}")

                            if not answer:
                                st.error("Gagal mendapatkan jawaban dari OpenAI. Periksa model, API key, dan quota/akses akun.")
                            else:
                                st.success(answer)

                except Exception as e:
                    st.error(f"Error saat memanggil OpenAI: {e}")

