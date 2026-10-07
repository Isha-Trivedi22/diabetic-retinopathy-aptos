import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG = 256
LABELS = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"]


@st.cache_resource
def get_model():
    return tf.keras.models.load_model("dr_model.keras")


# ---- same image-processing functions as the training notebook ----
def crop_black(img, tol=7):
    m = img.mean(axis=2) > tol
    if m.sum() == 0:
        return img
    ys, xs = np.where(m)
    return img[ys.min():ys.max(), xs.min():xs.max()]

def load_rgb(img, size=IMG):
    return cv2.resize(crop_black(img), (size, size))

def enhance(img, size=IMG):
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), size / 30), -4, 128)

def fov_mask(img, erode=12):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = (g > 15).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    h, w = m.shape
    ell = np.zeros_like(m)
    cv2.ellipse(ell, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, 255, -1)
    m = cv2.bitwise_and(m, ell)
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1)))

def optic_disc_mask(g, fov, r=28):
    blur = cv2.GaussianBlur(g, (0, 0), 12)
    blur[fov == 0] = 0
    _, _, _, loc = cv2.minMaxLoc(blur)
    m = np.zeros_like(g)
    cv2.circle(m, loc, r, 255, -1)
    return m

def keep_blobs(mask, min_area, max_area, max_aspect):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        asp = max(w, h) / max(1, min(w, h))
        if min_area <= a <= max_area and asp <= max_aspect:
            out[lab == i] = 255
    return out

def segment(img):
    fov = fov_mask(img)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(img[:, :, 1])
    od = optic_disc_mask(g, fov)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    d = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)

    def thr(x, pct):
        t = np.percentile(x[fov > 0], pct)
        return ((x > t) * 255).astype(np.uint8)

    b = thr(b, 97)
    d = thr(d, 96)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, sm)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, sm)
    b = cv2.bitwise_and(b, fov)
    d = cv2.bitwise_and(d, fov)
    b[od > 0] = 0
    d[od > 0] = 0
    return keep_blobs(b, 4, 400, 3.5), keep_blobs(d, 3, 150, 3.0)



st.set_page_config(page_title="DR Grading (APTOS 2019)", page_icon="👁️")
st.title("Diabetic Retinopathy Grading (APTOS 2019)")
st.caption("EfficientNetB0 classifier + classical image-processing lesion overlay. "
           "Educational project - NOT a medical diagnosis tool.")

file = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])
if file is not None:
    image = np.array(Image.open(file).convert("RGB"))
    img = load_rgb(image)
    x = enhance(img).astype(np.float32)[None]          # raw 0-255, same as training
    probs = get_model().predict(x, verbose=0)[0]
    bright, dark = segment(img)
    overlay = img.copy()
    overlay[bright > 0] = [255, 255, 0]                # yellow = bright lesion candidates
    overlay[dark > 0] = [255, 0, 0]                    # red = dark lesion candidates

    col1, col2 = st.columns(2)
    col1.image(img, caption="Input (cropped, resized)")
    col2.image(overlay, caption="Lesion candidates (yellow = bright, red = dark)")
    st.subheader("Predicted grade: " + LABELS[int(probs.argmax())])
    st.bar_chart({LABELS[i]: float(probs[i]) for i in range(5)})
