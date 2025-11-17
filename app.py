#!/usr/bin/env python3
"""
Streamlit app (refactored):
- Vehicle Detector + GPT-4o-mini chatbot
- Modular structure: config, model loading, detection, visualization, chatbot
- Lazy imports for heavy libs (ultralytics, cv2)
- Uses streamlit caching where appropriate
- Replaced deprecated `use_column_width` with `use_container_width`
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np
import requests

# ---------- Constants ----------
DEFAULT_DATASET_PATH = "/content/dataset"
DEFAULT_MODEL_PATH = "models/model_best.pt"
SESSION_LAST_DETECTIONS_KEY = "last_detections"

# ---------- Page config ----------
st.set_page_config(layout="wide", page_title="Vehicle Detector + GPT-4o-mini")

# ---------- Helper: Sidebar / config ----------
def init_sidebar() -> Dict:
    st.sidebar.header("Configuration")
    dataset_path = st.sidebar.text_input("Dataset path (optional)", DEFAULT_DATASET_PATH)
    model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    show_boxes = st.sidebar.checkbox("Show bounding boxes", True)
    api_key_input = st.sidebar.text_input("OpenAI API Key (optional)", os.getenv("OPENAI_API_KEY", ""), type="password")
    use_small_model = st.sidebar.checkbox("Use small demo model (yolov8n)", False)

    OPENAI_API_KEY = (api_key_input.strip() if api_key_input else os.getenv("OPENAI_API_KEY", "")).strip()
    MODEL_URL = os.getenv("MODEL_URL", "").strip()

    # ensure model parent dir exists
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    return {
        "dataset_path": dataset_path,
        "model_path": model_path,
        "conf_thresh": conf_thresh,
        "show_boxes": show_boxes,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "MODEL_URL": MODEL_URL,
        "use_small_model": use_small_model,
    }

# ---------- Utilities ----------
def download_model(url: str, dest: str, timeout: int = 120) -> None:
    """Download model from URL to dest (streaming). Raises on failure."""
    if not url:
        raise ValueError("MODEL_URL kosong.")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    tmp = dest + ".download"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)

def draw_boxes_cv2(img_pil: Image.Image, detections: List[dict], names: Dict[int, str]) -> Image.Image:
    """Draw bounding boxes using cv2 if available, otherwise return original PIL image."""
    try:
        import cv2
    except Exception:
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
            cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(img)

# ---------- Model loading (lazy + cached) ----------
@st.cache_resource
def load_yolo_model(path: str, small_demo: bool = False):
    """
    Lazy import & load YOLO model.
    Returns tuple (model, names_dict).
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"ultralytics import gagal: {e}")

    if small_demo:
        model = YOLO("yolov8n.pt")
    else:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model tidak ditemukan di {path}")
        model = YOLO(str(path))

    # Try to get class names
    names = {}
    try:
        names = getattr(model, "names", {}) or {}
    except Exception:
        names = {}
    return model, names

# ---------- Detection pipeline ----------
def run_detection(model, img: Image.Image, conf_thresh: float) -> List[dict]:
    """
    Run model.predict on a PIL image and return list of detections:
    each detection: {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}
    """
    res = model.predict(source=np.array(img), conf=conf_thresh, imgsz=640, verbose=False)
    r = res[0]
    detections = []
    if hasattr(r, "boxes") and r.boxes is not None:
        for b in r.boxes:
            # extract xyxy
            xy = []
            try:
                xy = b.xyxy.cpu().numpy().flatten().tolist()
            except Exception:
                try:
                    xy = b.xyxy.tolist()
                except Exception:
                    xy = []
            # confidence
            try:
                conf = float(b.conf.cpu().numpy())
            except Exception:
                conf = float(getattr(b, "conf", 0.0))
            # class id
            try:
                cls = int(b.cls.cpu().numpy())
            except Exception:
                cls = int(getattr(b, "cls", -1))

            if xy and conf >= conf_thresh:
                detections.append({"bbox": xy, "conf": conf, "class_id": cls})
    return detections

# ---------- UI Rendering helpers ----------
def render_input_column() -> List[Tuple[str, Image.Image]]:
    st.header("Input")
    uploaded = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    imgs = []
    if uploaded:
        for u in uploaded:
            try:
                imgs.append((u.name, Image.open(u).convert("RGB")))
            except Exception as e:
                st.error(f"Gagal membuka file {u.name}: {e}")
    return imgs

def render_detection_results_panel(imgs: List[Tuple[str, Image.Image]], cfg: Dict):
    st.header("Detection Results")
    if not imgs:
        st.info("Unggah gambar untuk menjalankan deteksi.")
        return

    model_file = cfg["model_path"] if not cfg["use_small_model"] else None

    # ensure model exists or download
    if not cfg["use_small_model"] and not Path(model_file).exists():
        if cfg["MODEL_URL"]:
            with st.spinner("Model tidak ditemukan – mengunduh dari MODEL_URL ..."):
                try:
                    download_model(cfg["MODEL_URL"], model_file)
                    st.success("Model selesai diunduh.")
                except Exception as e:
                    st.error(f"Gagal mengunduh model dari MODEL_URL: {e}")
                    st.stop()
        else:
            st.error("Model tidak ada dan MODEL_URL tidak di-set.")
            st.stop()

    # load model
    try:
        model, names = load_yolo_model(model_file if model_file else "", small_demo=cfg["use_small_model"])
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # process each image
    for fname, img in imgs:
        st.subheader(fname)
        try:
            detections = run_detection(model, img, cfg["conf_thresh"])
        except Exception as e:
            st.error(f"Gagal menjalankan prediksi pada {fname}: {e}")
            continue

        counts: Dict[str, int] = {}
        for d in detections:
            cname = names.get(d["class_id"], str(d["class_id"]))
            counts[cname] = counts.get(cname, 0) + 1

        if cfg["show_boxes"]:
            vis = draw_boxes_cv2(img, detections, names)
            # patched: use_container_width
            st.image(vis, use_container_width=True)

        st.write("Counts:", counts)
        # save to session for chatbot use
        st.session_state[SESSION_LAST_DETECTIONS_KEY] = {"file": fname, "detections": detections, "counts": counts}

# ---------- Chatbot panel ----------
def chat_panel(openai_api_key: str):
    st.markdown("---")
    st.header("Chat with Detector (GPT-4o-mini)")
    if SESSION_LAST_DETECTIONS_KEY not in st.session_state:
        st.info("Jalankan deteksi dulu sebelum chat.")
        return

    user_q = st.text_input("Tanyakan sesuatu terkait hasil deteksi (contoh: \"Ada berapa mobil?\")", key="chat_q")
    if st.button("Tanya GPT-4o-mini"):
        if not openai_api_key:
            st.error("OPENAI API key kosong. Masukkan di sidebar atau set OPENAI_API_KEY sebagai Secret.")
            return

        det_info = st.session_state[SESSION_LAST_DETECTIONS_KEY]
        prompt = (
            "Kamu adalah asisten AI yang membantu menjelaskan hasil deteksi kendaraan.\n"
            f"Berikut data deteksi terakhir:\n{json.dumps(det_info, indent=2)}\n"
            f"Pertanyaan pengguna: {user_q}\n"
            "Jawablah singkat dan informatif dalam Bahasa Indonesia."
        )

        with st.spinner("Menghubungi OpenAI..."):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)

                # prefer chat.completions if available, fallback to responses
                answer = None
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
                    if hasattr(resp, "choices") and len(resp.choices) > 0:
                        choice = resp.choices[0]
                        msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
                        if msg:
                            answer = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                        else:
                            answer = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
                except Exception:
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

# ---------- Main ----------
def main():
    st.title("🚗 Vehicle Detector + GPT-4o-mini Chatbot (Capstone 4)")

    cfg = init_sidebar()
    # add MODEL_URL to cfg for convenience
    cfg["MODEL_URL"] = cfg.get("MODEL_URL", os.getenv("MODEL_URL", "").strip())

    # Layout: left = input, right = detection results
    col1, col2 = st.columns([1, 1])
    with col1:
        imgs = render_input_column()
    with col2:
        render_detection_results_panel(imgs, cfg)

    # Chat panel (full width)
    chat_panel(cfg["OPENAI_API_KEY"])

if __name__ == "__main__":
    main()
