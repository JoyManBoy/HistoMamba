# HistoMamba: Spatial Mamba for Histopathology Image Classification

This repository contains the PyTorch implementation for `HistoMamba`, a deep learning model for patch-based classification of histopathology whole-slide images. It leverages a Spatial Mamba Block, which combines the sequence modeling capabilities of Mamba with a spatial aggregation mechanism for 2D image analysis.

## Features

- **Mamba-based Architecture**: Uses `SpatialMambaBlock` for efficient feature extraction from high-resolution image patches.
- **Modular Codebase**: Code is organized into logical modules for data handling, model architecture, and training logic.
- **Efficient Training**:
  - Automatic Mixed Precision (AMP) for faster training and lower VRAM usage.
  - Weighted Random Sampler to handle class imbalance.
  - Manifest Caching to speed up dataset initialization on subsequent runs.
- **Regularization**: Includes reconstruction loss, stochastic depth (DropPath), and standard data augmentations.
- **Monitoring**: Integrated with TensorBoard for live monitoring of training and validation metrics.
- **Early Stopping**: Stop training when the validation metric ceases to improve.

## Project Structure

```
histomamba/
├── README.md                 # This file
├── requirements.txt          # Package dependencies
├── train.py                  # Main script to run training
└── src/
    ├── data/
    │   ├── dataset.py        # HistoPatchDataset class
    │   └── utils.py          # Data helper functions
    ├── engine/
    │   └── trainer.py        # train_epoch and validate_epoch loops
    └── models/
        └── histomamba.py     # HistoMambaDeep model definition
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JoyManBoy/HistoMamba.git
cd histomamba
```

### 2. Install Dependencies

It's highly recommended to use a virtual environment (e.g., `conda` or `venv`).

```bash
# Create and activate a conda environment (recommended)
conda create -n histomamba python=3.10
conda activate histomamba

# Install PyTorch (check https://pytorch.org/ for the correct command for your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Mamba and other requirements
# The official mamba-ssm package requires specific dependencies
pip install mamba-ssm causal-conv1d>=1.1.0
pip install -r requirements.txt
```

### 3. Data Preparation

The code expects a specific data structure:

```
<data_dir>/
├── <sample_id_1>/
│   └── patches/
│       └── <sample_id_1>_patches.h5
├── <sample_id_2>/
│   └── patches/
│       └── <sample_id_2>_patches.h5
└── ...
```
- `<data_dir>` is the main directory you provide with the `--data_dir` argument.
- Each sample (e.g., a whole-slide image) should have its own folder named with its unique ID (`<sample_id_1>`).
- Inside each sample folder, there should be a subdirectory named `patches` (or as specified by `--patch_dir_name`).
- This subdirectory contains a single `.h5` file with the extracted image patches. The HDF5 dataset key for the patches is assumed to be `data`.

## Usage

Run the `train.py` script with your desired arguments.

### Example Command

```bash
python train.py \
    --data_dir /path/to/your/data \
    --metadata_csv /path/to/your/metadata.csv \
    --output_dir ./histomamba_output \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --lambda_recon 0.05 \
    --use_weighted_sampler \
    --monitor_metric val_f1 \
    --early_stopping_patience 10
```

### Key Arguments

- `--data_dir`: (Required) Path to your main data directory.
- `--metadata_csv`: Path to a CSV file containing sample IDs and labels. Must have an `id` column and a label column (e.g., `tissue_type`).
- `--output_dir`: Where to save checkpoints, logs, and caches.
- `--use_weighted_sampler`: Use to balance classes during training.
- `--no_amp`: Disable Automatic Mixed Precision.
- `--monitor_metric`: Metric to watch for saving the best model and for early stopping (`val_loss`, `val_accuracy`, `val_f1`).

For a full list of options, run:
```bash
python train.py --help
```

## Output

The script will generate the following in your `--output_dir`:
- **TensorBoard Logs**: Can be viewed with `tensorboard --logdir ./histomamba_output`.
- **`best_model_patch.pth`**: A checkpoint of the model with the best validation score.
- **`manifest_cache/`**: Cached file lists to speed up dataset loading.
