# Capstone4 - Vehicle Detector (Full Project)

This repo contains end-to-end code for training, evaluating, and deploying a vehicle detection model (YOLOv8) + Streamlit app with GPT-4o-mini chatbot.

## Included files
- train.py — training pipeline using ultralytics YOLO API
- evaluation.py — COCO-style evaluation (mAP) using ultralytics.YOLO.val()
- app.py — Streamlit app with GPT-4o-mini chatbot integration
- requirements.txt

## Dataset structure expected 
vehicle-detection.v1i.yolov12/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
  data.yaml


