import argparse
from pathlib import Path

import numpy as np


def normalize_per_channel(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-6
    return (X - mean) / std


def main():
    parser = argparse.ArgumentParser(description="Minimal preprocessing: normalize and optionally transpose.")
    parser.add_argument("--input", required=True, help="Path to raw X numpy file (shape [N, C, T] or [N, T, C]).")
    parser.add_argument("--labels", required=True, help="Path to labels numpy file (shape [N]).")
    parser.add_argument("--output-dir", required=True, help="Directory to write X.npy and y.npy.")
    parser.add_argument("--transpose", action="store_true", help="Transpose from [N, T, C] to [N, C, T].")
    args = parser.parse_args()

    X = np.load(args.input)
    y = np.load(args.labels)
    if args.transpose:
        X = np.asarray(X).transpose(0, 2, 1)
    X = normalize_per_channel(np.asarray(X, dtype=np.float32))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    print(f"Saved normalized data to: {out_dir}")


if __name__ == "__main__":
    main()
