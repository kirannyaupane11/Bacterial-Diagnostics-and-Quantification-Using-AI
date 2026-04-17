# 🦠 Using AI in Bacterial Diagnostics and Quantification

> **COS6032-E Industrial AI Project** | University of Bradford | EP7 Group  
> **Team:** Kiran Nyaupane · Bandika Dhital · Jenish Dani  
> **Supervisor:** Dr Kulvinder Panesar | **Client:** Dr Maria Katsikogianni

---

##  Overview

This project develops an **end-to-end AI pipeline** for the automated detection, segmentation, and quantification of *Staphylococcus aureus* bacteria in fluorescence microscopy images.

Manual bacterial counting under a microscope is slow, inconsistent, and requires specialist expertise — causing diagnostic delays and contributing to antimicrobial resistance (AMR) in low- and middle-income countries (LMICs). This system replaces that process with a deep learning model that runs in seconds.

---

##  Results

| Metric | Score |
|--------|-------|
| Dice Coefficient (test) | **81.1%** |
| IoU Score | **68.2%** |
| Precision | **73.6%** |
| Recall | **90.6%** |
| Best Val Dice (epoch 150) | **82.4%** |
| Counting error reduction | **73%** (63 → 10 total errors) |

---

##  Dataset

- **DeepBacs** — *Staphylococcus aureus* segmentation dataset (Ouyang et al., 2022)
- JE2 strain · Nile Red fluorescent stain · `.tif` format · 256×256 px patches
- 28 training image-mask pairs + 5 unseen test images
- Open access — no patient data, GDPR compliant

> Dataset is **not included** in this repository. It must be placed in your Google Drive at:  
> `MyDrive/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/`

---

##  Pipeline

```
Fluorescence Microscopy Image
        ↓
  Data Cleaning (5-point check)
        ↓
  Preprocessing (256×256, normalise, binarise)
        ↓
  Augmentation (28 → 84 samples)
        ↓
  U-Net Training (150 epochs, T4 GPU)
        ↓
  Binary Mask Prediction (sigmoid threshold 0.5)
        ↓
  Watershed Counting Algorithm
        ↓
  Results + Streamlit Dashboard
```

---

##  Model Architecture — U-Net

- **Encoder:** 4 blocks (16 → 32 → 64 → 128 filters) + MaxPool + Dropout
- **Bottleneck:** 256 filters
- **Decoder:** 4 blocks with Conv2DTranspose + skip connection concatenation
- **Output:** Conv2D(1, sigmoid) — per-pixel probability map
- **Total parameters:** 1,940,817
- **Framework:** TensorFlow 2.19 / Keras
- **Loss:** Dice Loss + Binary Cross-Entropy
- **Optimiser:** Adam (lr=1e-4)

---

##  How to Run

### 1. Open in Google Colab

Upload `Bacterial_detection_and_quantification.ipynb` to Google Colab or open it directly from your Google Drive.

### 2. Mount Google Drive

Run **Cell 1** — this connects the notebook to your Drive where the dataset and model weights are stored.

### 3. Install Dependencies

Run **Cell 2**:

```bash
pip install tensorflow opencv-python-headless matplotlib scikit-learn tifffile numpy pillow scipy streamlit
```

### 4. Set Dataset Path

In **Cell 4**, update `BASE_PATH` to point to your dataset location on Google Drive:

```python
BASE_PATH = "/content/drive/MyDrive/DeepBacs_Data_Segmentation_Staph_Aureus_dataset/..."
```

### 5. Run All Cells in Order

Run cells 1–20 sequentially. The full pipeline — from data loading to trained model to live dashboard — takes approximately **15 minutes** on a T4 GPU.

### 6. Launch Dashboard

**Cell 19** starts the Streamlit app and generates a public URL via localtunnel:

```
🌐 Your URL is: https://bacteriaai.loca.lt
```

If a password is required, run **Cell 20** to retrieve the tunnel IP address.

---

##  Repository Structure

```
├── Bacterial_detection_and_quantification.ipynb   # Main pipeline notebook (20 cells)
├── app.py                                          # Streamlit dashboard (525 lines)
├── results_summary.json                            # Final evaluation results
├── README.md
└── /docs
    ├── Design_Implementation_Report.docx
    ├── Demo_Plan.docx
    └── Agile_PM_Framework.xlsx
```

---

##  Notebook Cell Summary

| Cell | Purpose | Key Output |
|------|---------|-----------|
| 1 | Mount Google Drive | Drive accessible |
| 2 | Install libraries | TF 2.19, OpenCV, Streamlit |
| 3 | Import libraries | All imports confirmed |
| 4 | Define dataset paths | 4 paths verified ✓ |
| 5 | Explore dataset | 28 train + 5 test pairs |
| 6 | Visualise samples | Images and masks plotted |
| 7 | Data cleaning | 28 valid pairs, 0 issues ✓ |
| 8 | Preprocessing | Arrays (28, 256, 256, 1) |
| 9 | Augmentation + split | 84 samples → 67 train / 17 val |
| 10 | Build U-Net | 1,940,817 parameters |
| 11 | Compile model | Adam + Dice+BCE loss |
| 12 | Train model | Best val Dice = 0.8235 at epoch 90 |
| 13 | Plot training curves | Dice and Loss charts saved |
| 14 | Generate predictions | Visualisation on test images |
| 15 | Evaluate | Dice 79.6%, Recall 83.3% |
| 16 | Watershed counting | Error: 64 → 14 (78% improvement) |
| 17 | Save results | results_summary.json |
| 18 | Build Streamlit app | app.py (525 lines) |
| 19 | Deploy dashboard | Live URL via localtunnel |
| 20 | Get tunnel password | IP address for authentication |

---

##  Streamlit Dashboard Features

- Upload any `.tif` / `.png` / `.jpg` microscopy image
- Automatic AI inference using the trained U-Net
- Three-panel display: original | predicted mask | teal overlay
- Bacteria count (Watershed), coverage %, confidence metrics
- Adjustable threshold slider (0.1–0.9)
- Export: JSON · CSV · Report PNG · Mask PNG
- Session history with bulk export
- Model Performance tab with training curves and counting charts

---

##  Tech Stack

| Tool | Version | Use |
|------|---------|-----|
| Python | 3.12 | Core language |
| TensorFlow / Keras | 2.19 | U-Net model |
| OpenCV | 4.13 | Image resizing |
| scikit-image | latest | Watershed algorithm |
| scikit-learn | 1.6 | Evaluation metrics, train/test split |
| tifffile | latest | Loading .tif microscopy images |
| NumPy / SciPy | 2.0 / 1.16 | Arrays, distance transform |
| Matplotlib | 3.10 | Visualisations |
| Streamlit | 1.56 | Interactive dashboard |
| localtunnel | latest | Public URL for live demo |
| Google Colab | — | Cloud GPU (T4) training |

---

##  Ethical Considerations

- **Research prototype only** — not validated for clinical deployment
- No patient data used — DeepBacs is a publicly available laboratory dataset
- Model trained on a single bacterial strain and protocol — generalisation to other species or microscopes is not guaranteed
- AI should assist, not replace, clinician judgement


---

##  Team Members

| Name | Role |
|------|------|
| Kiran Nyaupane | ScrumMaster · Model Development · Dashboard |
| Bandika Dhital | Data Engineering · Preprocessing · Augmentation |
| Jenish Dani | Testing · Evaluation · Watershed Counting |

**Supervisor:** Dr Kulvinder Panesar, University of Bradford  
**Client:** Dr Maria Katsikogianni, University of Bradford

---

*University of Bradford · Faculty of Engineering and Informatics · COS6032-E Industrial AI Project · 2025–2026*
