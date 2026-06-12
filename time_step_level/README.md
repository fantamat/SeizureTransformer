# Time-Step Level Seizure Detection

This directory contains the implementation for **time-step level seizure detection** using the SeizureTransformer model. The model performs sequence-to-sequence modeling, directly outputting predictions at the time-step level without requiring redundant overlapping inferences.

## � Quick Reference

| Task | Command | RAM Usage |
|------|---------|-----------|
| **Prepare Dataset (Chunked)** | `python get_dataset_chunked.py --data-siena /path --data-ethz /path` | Low (~8GB) |
| **Prepare Dataset (Legacy)** | `python get_dataset.py --data-siena /path --data-ethz /path` | High (~100GB) |
| **Train (Chunked)** | `python train_sd.py --use_chunked --batch_size 86` | Low (~12GB) |
| **Train (Legacy)** | `python train_sd.py --batch_size 86` | High (~120GB) |
| **Evaluate Model** | `python eval_test.py` | Low |
| **Convert to Chunked** | `python dataset_utils.py convert --data-path ... --label-path ...` | Medium |
| **Verify Dataset** | `python dataset_utils.py verify metadata.json` | Low |

**💡 Tip:** Use chunked format for systems with limited RAM (<64GB). It provides ~90% RAM reduction with minimal performance impact.

## �📋 Table of Contents

- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Memory-Efficient Processing](#memory-efficient-processing)
- [Complete Workflow](#complete-workflow)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The SeizureTransformer is a U-shaped model that efficiently learns representations by capturing both local and global features using convolution and self-attentive modules. It won **1st place** in the 2025 seizure detection challenge.

**Key Features:**
- Direct time-step level predictions (no post-processing needed)
- Memory-efficient chunked dataset support
- Multi-dataset training (Siena, TUH EEG, CHB-MIT)
- State-of-the-art performance with cross-subject generalization

## 📦 Dataset Preparation

### Required Datasets

1. **Siena Scalp EEG Database**
   - Download: [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
   - Save to: `./data/BIDS_Siena` or specify with `--data-siena`

2. **TUH EEG Seizure Corpus v2.0.3**
   - Download: [TUSZ v2.0.3](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tueg)
   - Save to: `./data/v2.0.3/edf/train` or specify with `--data-ethz`

3. **CHB-MIT Scalp EEG Database** (Optional)
   - Download: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)
   - Save to: `./data/BIDS_CHB-MIT` or specify with `--data-mit`

### Dataset Preparation Options

You have two options for preparing datasets:

#### Option A: Legacy Format (Simple but High RAM Usage)

Process all data into single `.npy` files:

```bash
python get_dataset.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 1.0 \
    --data-siena /path/to/siena \
    --data-ethz /path/to/ethz
```

**Arguments:**
- `--window_size`: Input window size in samples (default: 15360 = 60 seconds at 256 Hz)
- `--alpha`: Ratio of full-seizure windows to partial-seizure windows (default: 0.7)
- `--beta`: Ratio of no-seizure windows to partial-seizure windows (default: 1.0)
- `--data-siena`: Path to Siena dataset directory
- `--data-ethz`: Path to TUH EEG dataset directory
- `--data-mit`: Path to CHB-MIT dataset directory (optional)

**Output:** Creates `.npy` files in `./data/dataset/`:
- `full_train_data_{alpha}_{beta}_{window_size}.npy`
- `full_train_label_{alpha}_{beta}_{window_size}.npy`

**⚠️ Warning:** This method can consume 100+ GB of RAM for large datasets.

#### Option B: Chunked Format (Recommended - Low RAM Usage)

Process data in memory-efficient chunks:

```bash
python get_dataset_chunked.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --chunk_size_gb 4.0 \
    --data-siena /path/to/siena \
    --data-ethz /path/to/ethz
```

**Arguments:**
- `--window_size`: Input window size (default: 15360)
- `--alpha`: Full-seizure to partial-seizure ratio (default: 0.7)
- `--beta`: No-seizure to partial-seizure ratio (default: 2.0)
- `--chunk_size_gb`: Maximum chunk size in GB (default: 4.0)
- `--data-siena`: Path to Siena dataset
- `--data-ethz`: Path to TUH EEG dataset
- `--data-mit`: Path to CHB-MIT dataset (optional)

**Output:** Creates chunked files in `./data/dataset/chunks/`:
- `full_train_{alpha}_{beta}_{window_size}_metadata.json`
- `full_train_{alpha}_{beta}_{window_size}_chunk0000_data.npy`
- `full_train_{alpha}_{beta}_{window_size}_chunk0000_label.npy`
- ... (additional chunks as needed)

**✅ Benefits:** Reduces RAM usage by ~90% (from 100+ GB to 8-12 GB during processing)

## 🚀 Training

### Training with Legacy Format

```bash
python train_sd.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --batch_size 86 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 2e-5
```

### Training with Chunked Format (Recommended)

```bash
python train_sd.py \
    --use_chunked \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --batch_size 86 \
    --chunk_cache_size 2 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 2e-5
```

### Training Arguments

**Model Architecture:**
- `--num_channel`: Number of EEG channels (default: 18)
- `--dim_feedforward`: Dimension of feedforward network (default: 2048)
- `--num_layers`: Number of transformer layers (default: 8)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.1)

**Dataset Configuration:**
- `--window_size`: Input sequence length (default: 15360)
- `--alpha`: Full-seizure ratio (default: 0.7)
- `--beta`: No-seizure ratio (default: 2.0)
- `--threshold`: Binary classification threshold for evaluation (default: 0.8)

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 86)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for regularization (default: 2e-5)
- `--lradj`: Learning rate adjustment strategy (default: 'type1')

**Chunked Dataset Options:**
- `--use_chunked`: Enable chunked dataset loading (flag)
- `--chunk_cache_size`: Number of chunks to cache in memory (default: 2)

**GPU Configuration:**
- `--use_gpu`: Use GPU for training (default: True)
- `--gpu`: GPU device ID (default: 0)
- `--use_multi_gpu`: Use multiple GPUs (flag, default: True)
- `--devices`: GPU device IDs, comma-separated (default: '0,1')

**Output:** Model checkpoint saved to:
```
./ckp/full_18_beta{beta}_alpha{alpha}_window{window_size}_dim{dim_feedforward}_layer{num_layers}_num_head{num_heads}_f1.pth
```

## 📊 Evaluation

Evaluate trained model on test set:

```bash
python eval_test.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --threshold 0.8 \
    --dim_feedforward 2048 \
    --num_layers 8 \
    --num_heads 4
```

**Arguments:** Same as training, used to locate the correct checkpoint file.

**Output:** Prints sample-level and event-level metrics:
- F1 score
- Sensitivity (recall)
- Precision
- False positive rate

The evaluation script automatically loads the corresponding model checkpoint and evaluates on the test set located at `./data/v2.0.3/edf/eval`.

## 💾 Memory-Efficient Processing

For systems with limited RAM, use the chunked dataset system:

### Dataset Utilities

The `dataset_utils.py` script provides tools for managing chunked datasets:

#### Convert Legacy Dataset to Chunked Format

```bash
python dataset_utils.py convert \
    --data-path ./data/dataset/full_train_data_0.7_2.0_15360.npy \
    --label-path ./data/dataset/full_train_label_0.7_2.0_15360.npy \
    --output-dir ./data/dataset/chunks \
    --dataset-name full_train_0.7_2.0_15360 \
    --chunk-size-gb 4.0
```

#### Inspect Dataset

View dataset structure and statistics:

```bash
python dataset_utils.py inspect \
    ./data/dataset/chunks/full_train_0.7_2.0_15360_metadata.json
```

#### Verify Dataset Integrity

Check for corrupted or missing files:

```bash
python dataset_utils.py verify \
    ./data/dataset/chunks/full_train_0.7_2.0_15360_metadata.json
```

#### Test Loading Performance

Benchmark data loading speed:

```bash
python dataset_utils.py test \
    ./data/dataset/chunks/full_train_0.7_2.0_15360_metadata.json \
    --num-samples 10
```

### RAM Usage Comparison

**Legacy Format:**
- Dataset size: ~50,000 samples × 18 channels × 15,360 samples
- RAM required: ~106 GB (includes numpy overhead)

**Chunked Format (4GB chunks):**
- Same dataset
- RAM required: ~8-12 GB (configurable via `--chunk_cache_size`)
- **Reduction: ~90%**

### Configuration Tips

**For Low Memory Systems:**
```bash
--chunk_cache_size 1  # Keep only 1 chunk in memory
--batch_size 32       # Smaller batch size
```

**For High Memory Systems:**
```bash
--chunk_cache_size 4  # Keep 4 chunks in memory
--batch_size 128      # Larger batch size for faster training
```

## 🔄 Complete Workflow

### 1. Prepare Datasets (Chunked - Recommended)

```bash
python get_dataset_chunked.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --chunk_size_gb 4.0 \
    --data-siena /path/to/siena \
    --data-ethz /path/to/ethz
```

### 2. Verify Dataset (Optional but Recommended)

```bash
python dataset_utils.py verify \
    ./data/dataset/chunks/full_train_0.7_2.0_15360_metadata.json
```

### 3. Train Model

```bash
python train_sd.py \
    --use_chunked \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --batch_size 86 \
    --epochs 100 \
    --devices 0,1
```

### 4. Evaluate Model

```bash
python eval_test.py \
    --window_size 15360 \
    --alpha 0.7 \
    --beta 2.0 \
    --threshold 0.8
```

## 🏗️ Model Architecture

The SeizureTransformer (`model.py`) implements a U-shaped architecture with:

1. **Encoder Stack**: Multi-scale convolutional feature extraction
2. **Residual CNN Stack**: Deep residual connections for feature refinement
3. **Transformer Layers**: Self-attention mechanism for global context
4. **Decoder Stack**: Upsampling with skip connections
5. **Output Layer**: Time-step level binary classification

**Default Configuration:**
- Input: 18 channels × 15,360 samples (60 seconds at 256 Hz)
- Feedforward dimension: 2048
- Transformer layers: 8
- Attention heads: 4
- Output: Time-step level seizure probabilities

## 🔧 Troubleshooting

### Out of Memory Errors

**Problem:** Training crashes with OOM error.

**Solutions:**
1. Use chunked dataset: Add `--use_chunked` flag
2. Reduce `--chunk_cache_size` (try 1 or 2)
3. Reduce `--batch_size` (try 32 or 64)
4. Use fewer data loader workers

### Dataset Not Found

**Problem:** FileNotFoundError during training.

**Solution:** 
- For legacy: Run `get_dataset.py` first
- For chunked: Run `get_dataset_chunked.py` first
- Verify paths with `dataset_utils.py inspect`

### Slow Training

**Problem:** Training is slower than expected.

**Solutions:**
1. Increase `--chunk_cache_size` (if RAM allows)
2. Increase `--batch_size`
3. Ensure data is on local SSD (not network storage)
4. Use multiple GPUs: `--devices 0,1,2,3`

### Model Checkpoint Not Found

**Problem:** eval_test.py cannot find model checkpoint.

**Solution:** Ensure training arguments match:
- Same `--alpha`, `--beta`, `--window_size`
- Same `--dim_feedforward`, `--num_layers`, `--num_heads`

### Data Processing Takes Too Long

**Problem:** get_dataset.py runs out of memory or takes hours.

**Solution:** Use `get_dataset_chunked.py` instead - it processes incrementally without loading everything into RAM.

## 📁 File Structure

```
time_step_level/
├── README.md                      # This file
├── model.py                       # SeizureTransformer architecture
├── train_sd.py                    # Training script
├── eval_test.py                   # Evaluation script
├── get_dataset.py                 # Legacy dataset preparation
├── get_dataset_chunked.py         # Memory-efficient dataset preparation
├── chunked_dataset.py             # Chunked dataset implementation
├── dataset_utils.py               # Dataset management utilities
├── example_chunked_usage.py       # Example demonstrating chunked system
├── CHUNKED_DATASET_README.md      # Detailed documentation on chunked system
├── service/                       # Helper modules
│   ├── handle_data.py            # Data loading utilities
│   ├── post_process.py           # Post-processing functions
│   └── result.py                 # Evaluation metrics
├── ckp/                          # Model checkpoints (created during training)
└── data/                         # Dataset directory
    ├── BIDS_Siena/               # Siena dataset
    ├── v2.0.3/edf/               # TUH EEG dataset
    │   ├── train/                # Training data
    │   ├── dev/                  # Validation data
    │   └── eval/                 # Test data
    └── dataset/                  # Processed datasets
        ├── chunks/               # Chunked format (recommended)
        └── *.npy                 # Legacy format (single files)
```

## 📚 Additional Resources

- **Main Project README**: `../README.md`
- **Chunked Dataset System**: `CHUNKED_DATASET_README.md`
- **Paper**: [arXiv:2504.00336](https://arxiv.org/abs/2504.00336)

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{wu2025large,
  title={Large EEG-U-Transformer for Time-Step Level Detection Without Pre-Training},
  author={Wu, Kerui and Zhao, Ziyue and Yener, B{\"u}lent},
  journal={arXiv preprint arXiv:2504.00336},
  year={2025}
}
```

## 🏆 Competition Results

This model won **1st place** in the 2025 seizure detection challenge organized at the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders.

**Docker Image Available:**
```bash
docker pull yujjio/seizure_transformer
```
