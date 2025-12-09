import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_classifier(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = correct / total if total else 0.0
    return acc, np.array(all_labels), np.array(all_preds)


def report_and_confusion(labels: np.ndarray, preds: np.ndarray, class_names=None) -> Dict:
    cm = confusion_matrix(labels, preds)
    rep = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    return {"confusion_matrix": cm.tolist(), "classification_report": rep}
