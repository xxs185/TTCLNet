# TTCLNet
<img width="2950" height="1110" alt="image" src="https://github.com/user-attachments/assets/f7420ce5-4420-4da9-8300-45da3e3098f1" />
In this study, four common deep learning architectures were used: TCN, Transformer, 1D-CNN, and Bi-LSTM. Preprocessed sEMG and IMU signals were fed into the TCN-Transformer and 1D-CNN branches to extract modality-specific features. The resulting feature vectors were concatenated and then processed by a bidirectional cross-attention mechanism, enabling deep fusion of cross-modal information by allowing sEMG and IMU signals to interact and highlight important features from each modality. The fused feature representations were then passed through the Bi-LSTM module, which captured long-term temporal dependencies in both directions. After this, two independent fully connected layers generated final predictions for action category and motion intensity. Motion intensity was characterized by the angular velocity metric, which could serve as a control command for the robot's rotational speed. The combination of TCN and Transformer captures local temporal structures and long-range dependencies within sEMG signals, while 1D-CNN extracts key motion features from IMU data. Overall, this fusion process—consisting of feature extraction, cross-attention fusion, Bi-LSTM temporal processing, and final prediction—improves action recognition accuracy and stabilizes motion intensity estimation by effectively merging complementary information from both modalities.
This folder contains a minimal, clean training pipeline focused on preprocessing and training four variants:

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


## Minimal preprocessing helper

`python -m open_source.preprocess --input X_raw.npy --labels y_raw.npy --output-dir ./processed --transpose` will normalize per-channel (z-score) and optionally transpose to `[N, C, T]`. Adjust or extend as needed for your raw data.

## Contact
If you have any question about this project, please feel free to contact lichang@stu.qut.edu.cn
