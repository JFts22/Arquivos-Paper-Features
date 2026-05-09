#!/usr/bin/env python3
"""
Four classifiers all consuming a 20-feature TSFEL vector:
  - Random Forest (sklearn)
  - SVM RBF      (sklearn, with subsampling)
  - CNN1D        (PyTorch, hyperparameter sweep)
  - LSTM         (PyTorch, hyperparameter sweep)

Each train_eval(...) returns dict with val/test accuracy + F1 weighted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


# ───────────────────────── Random Forest ─────────────────────────

def train_eval_rf(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    seed: int = 42,
) -> dict:
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_val_pred = rf.predict(X_val)
    y_te_pred = rf.predict(X_te)
    return {
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
        "test_f1_weighted": float(f1_score(y_te, y_te_pred, average="weighted")),
    }


# ───────────────────────── SVM ─────────────────────────

def train_eval_svm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    seed: int = 42,
    max_train: int = 30_000,
) -> dict:
    if X_tr.shape[0] > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_tr.shape[0], size=max_train, replace=False)
        X_tr_use, y_tr_use = X_tr[idx], y_tr[idx]
    else:
        X_tr_use, y_tr_use = X_tr, y_tr

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    clf.fit(X_tr_use, y_tr_use)
    y_val_pred = clf.predict(X_val)
    y_te_pred = clf.predict(X_te)
    return {
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
        "test_f1_weighted": float(f1_score(y_te, y_te_pred, average="weighted")),
    }


# ───────────────────────── PyTorch utilities ─────────────────────────

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(X: np.ndarray, y: np.ndarray, batch: int = 128, shuffle: bool = False):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=shuffle)


def _train_torch(
    model: nn.Module, train_loader, val_loader,
    lr: float, max_epochs: int = 30, patience: int = 5,
    device=None,
) -> nn.Module:
    device = device or _device()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    bad = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        # validation
        model.eval()
        v_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * yb.size(0)
                n += yb.size(0)
        v_loss /= max(n, 1)

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model


def _torch_predict(model: nn.Module, X: np.ndarray, batch: int = 256, device=None) -> np.ndarray:
    device = device or _device()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
            logits = model(xb)
            out.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


# ───────────────────────── CNN1D ─────────────────────────

class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.body = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(dim * 2, dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(),
        )
        flat = (n_features // 2) * (dim * 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x):
        # x: (B, n_features) → (B, 1, n_features)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.body(x)
        return self.head(x)


def train_eval_cnn(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_classes: int, dim: int, lr: float,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    n_feat = X_tr.shape[1]
    model = CNN1D(n_features=n_feat, n_classes=n_classes, dim=dim)
    train_loader = _make_loader(X_tr, y_tr, batch=128, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch=256, shuffle=False)
    model = _train_torch(model, train_loader, val_loader, lr=lr)

    y_val_pred = _torch_predict(model, X_val)
    y_te_pred = _torch_predict(model, X_te)
    return {
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
        "test_f1_weighted": float(f1_score(y_te, y_te_pred, average="weighted")),
    }


# ───────────────────────── LSTM ─────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        # x: (B, n_features) → (B, n_features, 1)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, (h, _) = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def train_eval_lstm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    n_classes: int, dim: int, lr: float,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    model = LSTMClassifier(n_classes=n_classes, hidden=dim)
    train_loader = _make_loader(X_tr, y_tr, batch=128, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch=256, shuffle=False)
    model = _train_torch(model, train_loader, val_loader, lr=lr)

    y_val_pred = _torch_predict(model, X_val)
    y_te_pred = _torch_predict(model, X_te)
    return {
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1_weighted": float(f1_score(y_val, y_val_pred, average="weighted")),
        "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
        "test_f1_weighted": float(f1_score(y_te, y_te_pred, average="weighted")),
    }
