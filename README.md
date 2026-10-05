# COMP9517: Computer Vision 2026 Term 2 - One Night Miracle

## Traditional methods

### Method 1 - BOVW + SIFT + SVMs
 
#### Prerequisites
 
```bash
numpy
pandas
opencv-python
scikit-learn
matplotlib
tqdm
```
 
#### How to run the code
 
The pipeline is contained in `bovw_sift_svm.ipynb`, which imports the local modules in the same folder (`config.py`, `dataset.py`, `sift_utils.py`, `codebook.py`, `encode.py`, `train_eval.py`) — keep these files together.
 
Place the two dataset zips in the same folder as the notebook (`traditional_bovw/`):
 
- `train_select.zip`  (40 train + 10 validation images per class)
- `test_select.zip`   (10 test images per class)
The notebook extracts them to `./data/` automatically on first run. Class count, per-class splits, and random seed are set in the config cell and can be edited there.
 
Run all cells in `bovw_sift_svm.ipynb` sequentially. It runs the full study in order: builds the class subset, extracts and caches RootSIFT descriptors, learns the visual vocabulary, runs the representation ablation (codebook size, hard/soft assignment, TF-IDF, spatial pyramid), evaluates the best combined configuration, tunes the SVM parameter `C`, and evaluates the frozen baseline and improved models on the test set.
 
Results (metrics, `C`-tuning tables, confusion matrices, and prediction examples) are saved under `traditional_bovw/traditional_bovw_results/`.

### Method 2 - Random Forests

#### Prerequisites

```bash
numpy
pandas
opencv-python
scikit-image
scikit-learn
joblib
matplotlib
seaborn
```

#### How to run the code

This notebook is designed to run in **Google Colab**, as it uses `google.colab.drive` to mount Google Drive.

Requires access to a Google Drive folder containing the pre-zipped dataset:

- `COMP9517/Data/train_select.zip`
- `COMP9517/Data/test_select.zip`

If running locally instead of in Colab, remove the `drive.mount(...)` and `!cp .../drive/...` cells, and point directly to local copies of `train_select.zip` / `test_select.zip`.

Run all cells in `Traditional_Random_Forest.ipynb` sequentially.

Intermediate features and trained models are saved to `COMP9517/Data/` on Google Drive as `.npy` and `.joblib` files respectively, so extraction/training does not need to be repeated across sessions.

## Deep learning-based methods

### Method 1 - EfficientNetV2

#### Prerequisites

```bash
datasets
matplotlib
numpy
pandas
pillow
safetensors
scikit-learn
torch
torchvision
transformers
```

#### How to run the code

Run the notebook with a CUDA-capable GPU where possible. From the project root, provide the extracted dataset in the following structure:

- `Data/train_select/`
- `Data/test_select/`

Run all cells in `efficientnetv2.ipynb` sequentially. Checkpoints, final models, metrics, and figures are saved under `EfficientNetV2_S_outputs/`; interrupted experiments resume from the latest available checkpoint by default.

### Method 2 - ConvNeXtV2

#### Prerequisites

```bash
albumentations
evaluate
matplotlib
numpy
torch
albumentations
datasets
pillow
sklearn
transformers
```

#### How to run the code

This notebook is designed to run in Google Colab using a T4 GPU runtime, as it provides the memory needed to handle image batches.

It also requires access to a Google Drive folder containing the pre-zipped dataset:

- `COMP9517/Data/train_select.zip`
- `COMP9517/Data/test_select.zip`

Run all cells in `convnextv2.ipynb` sequentially.
