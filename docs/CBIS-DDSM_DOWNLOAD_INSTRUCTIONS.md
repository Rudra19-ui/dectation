# CBIS-DDSM Dataset Download Instructions

## Overview
The CBIS-DDSM dataset is a large mammography dataset (>10GB) that requires special tools to download.

## Method 1: TCIA Download Manager (Recommended)

1. **Download TCIA Download Manager**:
   - Go to: https://wiki.cancerimagingarchive.net/display/NBIA/Download+Manager+User+Guide
   - Download the appropriate version for your OS

2. **Import the manifest file**:
   - Open TCIA Download Manager
   - Import: `dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia`

3. **Set download directory**:
   - Set to: `dataset4/cbisddsm_download`

4. **Start download**:
   - This may take several hours depending on your connection

## Method 2: Manual Download from TCIA Website

1. **Visit the CBIS-DDSM page**:
   - Go to: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM

2. **Download the dataset**:
   - Look for "Download" or "Data Access" section
   - Download the full dataset

3. **Extract to directory**:
   - Extract to: `dataset4/cbisddsm_download`

## Method 3: Using Python Scripts

If you have access to the NBIA Python client:

```bash
pip install nbia-data-retriever
nbia-data-retriever --manifest dataset4/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia --output dataset4/cbisddsm_download --include-annotations
```

## Method 4: Alternative Datasets

If CBIS-DDSM is too large, consider these alternatives:

1. **Mini-MIAS Dataset** (smaller, ~25MB):
   - Download from: http://peipa.essex.ac.uk/info/mias/
   - Contains 322 mammogram images

2. **INbreast Dataset** (medium size):
   - Requires registration at: https://www.breast-cancer-dataset.com/

## After Download

Once you have the dataset:

1. **Process the data**:
   ```bash
   python process_cbisddsm.py
   ```

2. **Train the model**:
   ```bash
   python train_model.py
   ```

## Dataset Information

- **CBIS-DDSM**: Curated Breast Imaging Subset of DDSM
- **Size**: >10GB
- **Format**: DICOM files
- **Labels**: Normal, Benign, Malignant
- **Views**: CC (craniocaudal) and MLO (mediolateral oblique)
- **Quality**: High-quality, professionally annotated

## Troubleshooting

- **Large download size**: Consider downloading during off-peak hours
- **Network issues**: Use a download manager with resume capability
- **Storage space**: Ensure you have at least 15GB free space
- **Processing time**: Converting DICOM to PNG may take 1-2 hours

## Next Steps

1. Download the dataset using one of the methods above
2. Run: `python process_cbisddsm.py`
3. Run: `python train_model.py`
4. Test the web app: `python -m streamlit run webapp/streamlit_app.py`
