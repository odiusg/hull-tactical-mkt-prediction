from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMWrapper:
    """Sklearn-compatible wrapper around a PyTorch LSTM for time-series regression.

    fit(X, y):
        X is time-ordered (N, features).  Builds sliding-window sequences of
        length `lookback`, trains the LSTM, then stores the last (lookback-1)
        training rows as context for the first prediction steps.

    predict(X_val):
        Prepends stored training tail so every val row gets a full-length
        context window.  No val-set leakage: tail comes from training only.

    Note: Optuna tuning with 30 trials × 5-fold is slow on CPU (~25 min+).
    Use --n-trials 10 --models lstm when running standalone.
    """

    needs_scaling: bool = True  # consumed by models.needs_scaling() via getattr

    def __init__(
        self,
        lookback: int = 21,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._net: _LSTMNet | None = None
        self._train_tail: np.ndarray | None = None
        self._n_features: int = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y) -> LSTMWrapper:
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, n_features = X.shape
        self._n_features = n_features
        lb = self.lookback

        # Store last (lb-1) training rows to warm-start context in predict().
        self._train_tail = (
            X[-(lb - 1):] if lb > 1
            else np.empty((0, n_features), dtype=np.float32)
        )

        if n < lb:
            self._net = None
            return self

        # Build sequences: X[i:i+lb] → y[i+lb-1]  (predict last row's target)
        n_seq = n - lb + 1
        seqs = np.stack([X[i : i + lb] for i in range(n_seq)])  # (n_seq, lb, features)
        targets = y[lb - 1:]                                     # (n_seq,)

        dataset = TensorDataset(
            torch.from_numpy(seqs).to(self._device),
            torch.from_numpy(targets).to(self._device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net = _LSTMNet(n_features, self.hidden_size, self.n_layers, self.dropout).to(self._device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        log_every = max(1, self.epochs // 5)
        net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(f"    [LSTM] epoch {epoch+1:3d}/{self.epochs}  loss={epoch_loss/len(loader):.6f}")

        self._net = net
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        m = len(X)

        if self._net is None:
            return np.ones(m, dtype=np.float32)

        lb = self.lookback
        tail = self._train_tail
        X_full = np.vstack([tail, X]) if len(tail) > 0 else X
        # X_full: (lb-1 + m, features) → m sequences each of length lb
        seqs = np.stack([X_full[j : j + lb] for j in range(m)])  # (m, lb, features)

        self._net.eval()
        with torch.no_grad():
            t = torch.from_numpy(seqs).to(self._device)
            preds = self._net(t).cpu().numpy()
        return preds
