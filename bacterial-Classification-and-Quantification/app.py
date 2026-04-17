# Install Streamlit if it's not already installed


import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

st.title("AI-Based Bacterial Diagnostic & Quantification")

# -------- Load model ----------
@st.cache_resource
def load_model():
    # UNet is already defined in the notebook's scope
    model = UNet()
    model.load_state_dict(torch.load("best_unet.pth",
                                     map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------- Upload image ----------
uploaded = st.file_uploader(
    "Upload Fluorescence Image",
    type=["png","jpg","tif"]
)

if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    st.image(img, caption="Original Image")

    # preprocessing
    img_norm = (img-img.min())/(img.max()-img.min()+1e-8)
    tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
        mask = (pred>0.5).float().numpy()[0,0]

    st.image(mask, caption="Predicted Bacteria Mask")

    # -------- Quantification ----------
    coverage = mask.mean()*100

    mask_u8 = (mask*255).astype(np.uint8)
    num_labels,_ = cv2.connectedComponents(mask_u8)
    count = num_labels-1

    st.subheader("Quantification Results")
    st.write(f"Bacterial Coverage: {coverage:.2f}%")
    st.write(f"Estimated Count: {count}")
