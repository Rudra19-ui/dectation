# Project Structure: Automated Breast Cancer Detection Using Mammogram Images

## 1. Data
- `data/` : Raw and processed mammogram images (CBIS-DDSM, MIAS, etc.)
- `notebooks/` : Jupyter notebooks for EDA, prototyping, and experiments

## 2. Source Code
- `src/`
  - `data/` : Data loading, preprocessing, augmentation
  - `train/` : Training scripts and utilities
- `models/` : Deep learning model architectures and saved weights
- `utils/` : Helper functions (metrics, Grad-CAM, report generation)

## 3. API & Deployment
- `api/` : Flask API for model serving
- `webapp/` : Streamlit frontend for uploading images and viewing results

## 4. Reports & Outputs
- `reports/` : Generated PDF diagnostic reports, saved images, and heatmaps

## 5. Config & Docs
- `requirements.txt` : Python dependencies
- `README.md` : Project overview and instructions
- `project_structure.md` : Directory and file structure

---

## Example Directory Tree

```
project/
├── api/
│   └── app.py
├── data/
│   ├── images/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── baseline_cnn.py
│   ├── transfer_learning.py
│   └── breast_cancer_detector.pt
├── notebooks/
│   └── EDA_and_Prototyping.ipynb
├── reports/
│   ├── diagnosis_YYYYMMDD_HHMM.pdf
│   ├── uploaded_image.png
│   └── heatmap.png
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   └── train/
│       └── train.py
├── utils/
│   ├── metrics.py
│   ├── gradcam.py
│   └── report.py
├── webapp/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
└── project_structure.md
``` 