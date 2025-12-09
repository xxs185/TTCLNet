# TTCLNet

This folder contains a minimal, clean training pipeline focused on preprocessing and training four variants:

- LSTM classifier (`lstm`)
- Temporal Convolutional Network (`tcn`)
- Dual-branch EMG+IMU (`dual`) with four interaction modes:
  - `none` baseline (no cross-modal attention)
  - `emg_to_imu` (EMG attends to IMU)
  - `imu_to_emg` (IMU attends to EMG)
  - `bidirectional` (both directions)
- Dual-branch EMG-only baseline (`dual_no_imu`)

## Data format

All training scripts expect NumPy arrays:

- Single-branch (LSTM/TCN): `X.npy` shaped `[N, T, C]` and `y.npy` shaped `[N]` with integer class ids.
- Dual-branch: `X_emg.npy` shaped `[N, C_emg, T]`, `X_imu.npy` shaped `[N, C_imu, T]` (for `dual_no_imu` you can point `X_imu.npy` to an empty array or pass `--imu-channels 0`), and `y_actions.npy` shaped `[N]` (class ids). If you also have gyro bins, provide `y_gyro_bins.npy` (integer bins) and pass `--gyro-bins`.

All labels should already be encoded as `0..num_classes-1`. Channel-first convention is used for EMG/IMU inputs. If your saved arrays are time-first `[N, T, C]`, pass `--transpose-input` to transpose automatically during loading.

## Quick start

```bash
# LSTM
python -m open_source.train --model lstm --data-root /path/to/data --num-classes 6

# TCN
python -m open_source.train --model tcn --data-root /path/to/data --num-classes 6

# Dual-branch EMG+IMU (four interaction modes)
python -m open_source.train --model dual --data-root /path/to/data --num-classes 6 --gyro-bins 3 --attention-mode none
python -m open_source.train --model dual --data-root /path/to/data --num-classes 6 --gyro-bins 3 --attention-mode emg_to_imu
python -m open_source.train --model dual --data-root /path/to/data --num-classes 6 --gyro-bins 3 --attention-mode imu_to_emg
python -m open_source.train --model dual --data-root /path/to/data --num-classes 6 --gyro-bins 3 --attention-mode bidirectional

# Dual-branch EMG-only baseline
python -m open_source.train --model dual_no_imu --data-root /path/to/data --num-classes 6 --imu-channels 0
```

### Common flags

- `--data-root`: directory containing the required `.npy` files.
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--dropout`: standard training hyperparameters.
- `--val-ratio`: validation split ratio (default 0.2).
- `--seed`: random seed (default 42).
- `--transpose-input`: transpose `X.npy` from `[N, T, C]` to `[N, C, T]` at load time (single-branch) or applies to both EMG/IMU for dual models.

### Outputs

Each run writes to `runs/<timestamp>_<model>/`:

- `model.pt` – trained weights
- `config.json` – training configuration
- `metrics.json` – final train/val accuracy and loss
- `classification_report.txt` – sklearn classification report
- `confusion_matrix.npy/.txt` – confusion matrix values

## Minimal preprocessing helper

`python -m open_source.preprocess --input X_raw.npy --labels y_raw.npy --output-dir ./processed --transpose` will normalize per-channel (z-score) and optionally transpose to `[N, C, T]`. Adjust or extend as needed for your raw data.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
