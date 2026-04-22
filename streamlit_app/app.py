import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import os

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ── Classes ──────────────────────────────────────────
our_classes = [
    "Tomato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Yellow Leaf Curl Virus",
    "Potato Healthy",
    "Potato Early Blight",
    "Pepper Bell Healthy",
    "Pepper Bell Bacterial Spot"
]

# ── Model Paths ──────────────────────────────────────
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Models ──────────────────────────────────────
@st.cache_resource
def load_models():
    with st.spinner("⏳ Loading models..."):
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        svm      = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        rf       = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        dl_model = load_model(os.path.join(MODELS_DIR, "efficientnet_final.h5"))
    return scaler, svm, rf, dl_model

# ── Preprocessing ────────────────────────────────────
def load_and_filter(img_bgr):
    img_filtered = cv2.medianBlur(img_bgr, 3)
    img_resized  = cv2.resize(img_filtered, (224, 224))
    return img_resized

def get_segmentation(img_bgr):
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask        = np.zeros(img_bgr.shape[:2], np.uint8)
    rect        = (10, 10, 204, 204)
    bgd_model   = np.zeros((1, 65), np.float64)
    fgd_model   = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, rect, bgd_model,
                fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask_binary = np.where(
        (mask == 2) | (mask == 0), 0, 1
    ).astype("uint8")
    result = img_rgb * mask_binary[:, :, np.newaxis]
    return mask_binary, result

def extract_color_features(segmented_img):
    features = []
    for i in range(3):
        channel     = segmented_img[:, :, i]
        leaf_pixels = channel[channel > 0]
        if len(leaf_pixels) == 0:
            features.extend([0.0, 0.0])
        else:
            features.append(np.mean(leaf_pixels))
            features.append(np.std(leaf_pixels))
    segmented_hsv = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        channel     = segmented_hsv[:, :, i]
        leaf_pixels = channel[channel > 0]
        if len(leaf_pixels) == 0:
            features.extend([0.0, 0.0])
        else:
            features.append(np.mean(leaf_pixels))
            features.append(np.std(leaf_pixels))
    return np.array(features)

def extract_texture_features(segmented_img):
    gray   = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)
    coords = np.where(gray > 0)
    if len(coords[0]) == 0:
        return np.zeros(8)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    if (y_max - y_min) < 2 or (x_max - x_min) < 2:
        return np.zeros(8)
    gray_cropped = gray[y_min:y_max, x_min:x_max]
    distances    = [1, 2]
    angles       = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm         = graycomatrix(gray_cropped, distances=distances,
                                angles=angles, levels=256,
                                symmetric=True, normed=True)
    features   = []
    properties = ["contrast", "correlation", "energy", "homogeneity"]
    for prop in properties:
        result = graycoprops(glcm, prop)
        features.append(result.mean())
        features.append(result.std())
    return np.array(features)

def extract_shape_features(segmented_img, mask_binary):
    contours, _ = cv2.findContours(
        mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return np.zeros(6)
    largest  = max(contours, key=cv2.contourArea)
    area     = cv2.contourArea(largest)
    if area < 100:
        return np.zeros(6)
    perimeter    = cv2.arcLength(largest, True)
    circularity  = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    x, y, w, h   = cv2.boundingRect(largest)
    aspect_ratio = w / float(h)
    extent       = area / (w * h) if (w * h) > 0 else 0
    hull         = cv2.convexHull(largest)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0
    return np.array([area, perimeter, circularity,
                     aspect_ratio, extent, solidity])

def extract_all_features(img_bgr):
    mask, result = get_segmentation(img_bgr)
    color        = extract_color_features(result)
    texture      = extract_texture_features(result)
    shape        = extract_shape_features(result, mask)
    return np.concatenate([color, texture, shape]), mask, result

# ── Main UI ──────────────────────────────────────────
def main():
    # Header
    st.title("🌿 Plant Disease Detection System")
    st.markdown("### ML vs Deep Learning Comparison")
    st.markdown("---")

    # Load models
    scaler, svm, rf, dl_model = load_models()
    st.success("✅ All models loaded and ready!")

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["All Models (Compare)", "SVM Only",
         "Random Forest Only", "EfficientNetB0 Only"]
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Accuracies")
    st.sidebar.markdown("🔵 **SVM:** 90.99%")
    st.sidebar.markdown("🟢 **Random Forest:** 84.84%")
    st.sidebar.markdown("🔴 **EfficientNetB0:** 99.02%")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌿 Supported Classes")
    for cls in our_classes:
        st.sidebar.markdown(f"• {cls}")

    # Upload
    st.markdown("### 📤 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Read image
        file_bytes    = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        img_bgr       = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_processed = load_and_filter(img_bgr)

        # Show images
        st.markdown("### 🖼️ Image Processing Pipeline")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📷 Original Image**")
            st.image(img_rgb, use_column_width=True)

        with st.spinner("🔄 Applying GrabCut segmentation..."):
            features, mask, segmented = extract_all_features(img_processed)

        with col2:
            st.markdown("**✂️ GrabCut Segmentation**")
            st.image(segmented.astype(np.uint8), use_column_width=True)

        with col3:
            st.markdown("**🎭 Binary Mask**")
            st.image((mask * 255).astype(np.uint8),
                     use_column_width=True, clamp=True)

        st.markdown("---")
        st.markdown("### 🤖 Model Predictions")

        # Scale features for ML
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Prepare image for DL
        img_dl = cv2.cvtColor(
            img_processed, cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        img_dl = preprocess_input(img_dl)
        img_dl = np.expand_dims(img_dl, axis=0)

        # ── Predictions ──────────────────────────────
        svm_pred, svm_proba = None, None
        rf_pred,  rf_proba  = None, None
        dl_pred,  dl_proba  = None, None

        col_svm, col_rf, col_dl = st.columns(3)

        # SVM
        if model_choice in ["All Models (Compare)", "SVM Only"]:
            with col_svm:
                st.markdown("#### 🔵 SVM")
                svm_pred  = svm.predict(features_scaled)[0]
                svm_proba = svm.predict_proba(features_scaled)[0]
                svm_conf  = svm_proba[svm_pred] * 100
                st.success(f"**{our_classes[svm_pred]}**")
                st.metric("Confidence", f"{svm_conf:.2f}%")
                st.markdown("**Top 3 Predictions:**")
                top3 = np.argsort(svm_proba)[::-1][:3]
                for idx in top3:
                    bar_val = int(svm_proba[idx] * 100)
                    st.write(f"• {our_classes[idx]}: {bar_val}%")
                    st.progress(bar_val)

        # Random Forest
        if model_choice in ["All Models (Compare)", "Random Forest Only"]:
            with col_rf:
                st.markdown("#### 🟢 Random Forest")
                rf_pred  = rf.predict(features_scaled)[0]
                rf_proba = rf.predict_proba(features_scaled)[0]
                rf_conf  = rf_proba[rf_pred] * 100
                st.success(f"**{our_classes[rf_pred]}**")
                st.metric("Confidence", f"{rf_conf:.2f}%")
                st.markdown("**Top 3 Predictions:**")
                top3 = np.argsort(rf_proba)[::-1][:3]
                for idx in top3:
                    bar_val = int(rf_proba[idx] * 100)
                    st.write(f"• {our_classes[idx]}: {bar_val}%")
                    st.progress(bar_val)

        # EfficientNetB0
        if model_choice in ["All Models (Compare)", "EfficientNetB0 Only"]:
            with col_dl:
                st.markdown("#### 🔴 EfficientNetB0")
                with st.spinner("Running DL inference..."):
                    dl_proba = dl_model.predict(img_dl, verbose=0)[0]
                dl_pred = np.argmax(dl_proba)
                dl_conf = dl_proba[dl_pred] * 100
                st.success(f"**{our_classes[dl_pred]}**")
                st.metric("Confidence", f"{dl_conf:.2f}%")
                st.markdown("**Top 3 Predictions:**")
                top3 = np.argsort(dl_proba)[::-1][:3]
                for idx in top3:
                    bar_val = int(dl_proba[idx] * 100)
                    st.write(f"• {our_classes[idx]}: {bar_val}%")
                    st.progress(bar_val)

        # ── Model Agreement ───────────────────────────
        if model_choice == "All Models (Compare)":
            st.markdown("---")
            st.markdown("### 🤝 Model Agreement")
            predictions = {}
            if svm_pred is not None:
                predictions["🔵 SVM"]           = our_classes[svm_pred]
            if rf_pred is not None:
                predictions["🟢 Random Forest"] = our_classes[rf_pred]
            if dl_pred is not None:
                predictions["🔴 EfficientNetB0"] = our_classes[dl_pred]

            if len(set(predictions.values())) == 1:
                st.success(
                    f"✅ All models agree: **{our_classes[svm_pred]}**"
                )
            else:
                st.warning("⚠️ Models disagree!")
                cols = st.columns(len(predictions))
                for i, (model_name, pred) in enumerate(predictions.items()):
                    cols[i].info(f"**{model_name}**\n\n{pred}")
                st.info(
                    "💡 Trust **EfficientNetB0** (99.02% accuracy) "
                    "for the final decision."
                )

        # ── Features Table ────────────────────────────
        st.markdown("---")
        with st.expander("🔬 View Extracted Features (26 features)"):
            feature_names = [
                "R_mean","R_std","G_mean","G_std","B_mean","B_std",
                "H_mean","H_std","S_mean","S_std","V_mean","V_std",
                "Contrast_mean","Contrast_std",
                "Correlation_mean","Correlation_std",
                "Energy_mean","Energy_std",
                "Homogeneity_mean","Homogeneity_std",
                "Area","Perimeter","Circularity",
                "Aspect_Ratio","Extent","Solidity"
            ]
            df = pd.DataFrame({
                "Feature": feature_names,
                "Value":   [f"{v:.4f}" for v in features],
                "Type":    (["Color"]*12 + ["Texture"]*8 + ["Shape"]*6)
            })
            st.dataframe(df, use_container_width=True)

    else:
        # Welcome screen
        st.info("👆 Please upload a leaf image to get started!")
        st.markdown("---")
        st.markdown("### 🌿 How It Works")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("**1️⃣ Upload**\n\nUpload any leaf image")
        col2.markdown("**2️⃣ Segment**\n\nGrabCut isolates the leaf")
        col3.markdown("**3️⃣ Extract**\n\n26 features extracted")
        col4.markdown("**4️⃣ Predict**\n\nML + DL models classify")

if __name__ == "__main__":
    main()