import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ArrayDataset, load_single_branch, load_dual_branch, split_dataset
from .models import LSTMClassifier, TCNClassifier, DualBranchModel
from .utils import evaluate_classifier, report_and_confusion, save_json, set_seed


def build_model(args, input_channels, imu_channels=0):
    if args.model == "lstm":
        return LSTMClassifier(
            input_size=input_channels,
            hidden_size=args.hidden,
            num_layers=args.layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
    if args.model == "tcn":
        return TCNClassifier(
            input_channels=input_channels,
            num_classes=args.num_classes,
            channels=args.tcn_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if args.model in {"dual", "dual_no_imu"}:
        gyro_bins = args.gyro_bins if args.gyro_bins and args.gyro_bins > 0 else None
        return DualBranchModel(
            emg_channels=input_channels,
            imu_channels=imu_channels if args.model == "dual" else 0,
            num_classes=args.num_classes,
            gyro_bins=gyro_bins,
            emg_tcn_channels=args.emg_tcn_channels,
            emg_kernel_size=args.emg_kernel_size,
            emg_dropout=args.dropout,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim,
            imu_cnn_channels=args.imu_cnn_channels,
            imu_kernel_size=args.imu_kernel_size,
            imu_dropout=args.dropout,
            lstm_hidden=args.hidden,
            lstm_layers=args.layers,
            attention_mode=args.attention_mode,
            attention_heads=args.attention_heads,
            attention_dropout=args.attention_dropout,
        )
    raise ValueError(f"Unknown model type: {args.model}")


def train_single_branch(args, device):
    data_root = Path(args.data_root)
    X, y = load_single_branch(data_root, args.transpose_input)
    input_channels = X.shape[1] if args.transpose_input or X.ndim == 3 and X.shape[1] <= X.shape[2] else X.shape[2]

    dataset = ArrayDataset(X, y)
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    model = build_model(args, input_channels=input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for X_batch, y_batch in progress:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            running_loss += loss.item() * y_batch.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / total if total else 0.0
        train_loss = running_loss / total if total else 0.0
        val_acc, labels, preds = evaluate_classifier(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f}")

    return model, {"train_acc": train_acc, "train_loss": train_loss}, labels, preds


def train_dual_branch(args, device):
    data_root = Path(args.data_root)
    X_emg, X_imu, y_actions, y_gyro = load_dual_branch(data_root, args.transpose_input)
    dataset_emg = torch.from_numpy(np.asarray(X_emg, dtype=np.float32))
    dataset_imu = torch.from_numpy(np.asarray(X_imu, dtype=np.float32))
    y_actions = torch.from_numpy(np.asarray(y_actions, dtype=np.int64))
    y_gyro = torch.from_numpy(np.asarray(y_gyro, dtype=np.int64)) if y_gyro is not None else None

    class DualDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(y_actions)

        def __getitem__(self, idx):
            emg = dataset_emg[idx]
            imu = dataset_imu[idx]
            y_action = y_actions[idx]
            yg = y_gyro[idx] if y_gyro is not None else torch.tensor(0, dtype=torch.long)
            return emg, imu, y_action, yg

    full_dataset = DualDataset()
    train_set, val_set = split_dataset(full_dataset, args.val_ratio, args.seed)

    def collate(batch):
        emg, imu, ya, yg = zip(*batch)
        return (
            torch.stack(emg),
            torch.stack(imu),
            torch.stack(ya),
            torch.stack(yg),
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda", collate_fn=collate)

    model = build_model(
        args,
        input_channels=dataset_emg.shape[1],
        imu_channels=dataset_imu.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
        for emg, imu, y_action, y_gyro_batch in progress:
            emg = emg.to(device)
            imu = imu.to(device)
            y_action = y_action.to(device)
            y_gyro_batch = y_gyro_batch.to(device)

            action_logits, gyro_logits = model(emg, imu)
            loss = loss_fn(action_logits, y_action)
            if gyro_logits is not None and (args.gyro_weight > 0):
                loss = loss + args.gyro_weight * loss_fn(gyro_logits, y_gyro_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(action_logits, dim=1)
            correct += (preds == y_action).sum().item()
            total += y_action.size(0)
            running_loss += loss.item() * y_action.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / total if total else 0.0
        train_loss = running_loss / total if total else 0.0
        val_acc, labels, preds = evaluate_dual(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} train_loss={train_loss:.4f}")

    return model, {"train_acc": train_acc, "train_loss": train_loss}, labels, preds


def evaluate_dual(model, loader, device):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emg, imu, y_action, _ in loader:
            emg = emg.to(device)
            imu = imu.to(device)
            y_action = y_action.to(device)
            action_logits, _ = model(emg, imu)
            preds = torch.argmax(action_logits, dim=1)
            total += y_action.size(0)
            correct += (preds == y_action).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_action.cpu().numpy())
    acc = correct / total if total else 0.0
    return acc, np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Unified training entry for LSTM, TCN, and dual-branch models.")
    parser.add_argument("--model", required=True, choices=["lstm", "tcn", "dual", "dual_no_imu"])
    parser.add_argument("--data-root", required=True, help="Directory containing .npy data files.")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--gyro-bins", type=int, default=0, help="Bins for gyro head (dual models).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size for LSTM and dual LSTM head.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transpose-input", action="store_true", help="Transpose X from [N, T, C] to [N, C, T] at load time.")

    # TCN specific
    parser.add_argument("--tcn-channels", type=int, nargs="+", default=[64, 128, 128, 128])
    parser.add_argument("--kernel-size", type=int, default=5)

    # Dual-branch specifics
    parser.add_argument("--emg-tcn-channels", type=int, nargs="+", default=[64, 128, 128])
    parser.add_argument("--emg-kernel-size", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--imu-cnn-channels", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--imu-kernel-size", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--gyro-weight", type=float, default=1.0, help="Weight for gyro loss in dual models.")
    parser.add_argument(
        "--attention-mode",
        choices=["none", "emg_to_imu", "imu_to_emg", "bidirectional"],
        default="emg_to_imu",
        help="Dual-branch interaction: none (baseline), EMG attends to IMU, IMU attends to EMG, or bidirectional.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in {"lstm", "tcn"}:
        model, train_stats, labels, preds = train_single_branch(args, device)
    else:
        model, train_stats, labels, preds = train_dual_branch(args, device)

    acc = (preds == labels).mean() if len(labels) else 0.0
    metrics = report_and_confusion(labels, preds)
    metrics["train_stats"] = train_stats
    metrics["val_accuracy"] = acc
    metrics["class_names"] = list(range(args.num_classes))

    torch.save(model.state_dict(), out_dir / "model.pt")
    save_json(vars(args), out_dir / "config.json")
    save_json(metrics, out_dir / "metrics.json")
    np.savetxt(out_dir / "confusion_matrix.txt", np.array(metrics["confusion_matrix"]), fmt="%d")
    print(f"Saved model and logs to: {out_dir}")


if __name__ == "__main__":
    main()
