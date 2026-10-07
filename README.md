---
title: DR Grading APTOS
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
---

# Diabetic Retinopathy Grading (APTOS 2019)

EfficientNetB0 transfer-learning classifier (grades 0-4) with a classical image-processing
lesion overlay (green channel, CLAHE, top-hat / black-hat morphology).

Dataset: APTOS 2019 Blindness Detection (Kaggle). Educational project, not a medical device.
