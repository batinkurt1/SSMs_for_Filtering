from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasources.build_dense_targets import build_dense_targets
from datasources.normalization import normalize_y_u
from helpers.load_pickle import load_pickle

class HorizonWindowDataset(Dataset):
    """Dataset for sliding-window or full-context multi-horizon forecasting."""

    def __init__(
        self,
        measurements,
        inputs,
        window_L,
        horizon_H,
        dataset_type,
    ):
        super().__init__()
        self.dataset_type = dataset_type

        self.y = np.asarray(measurements, dtype=np.float32)
        
        self.u = None if inputs is None else np.asarray(inputs, dtype=np.float32)
        self.m = 0 if self.u is None else int(self.u.shape[-1])

        self.H = int(horizon_H)

        N, Tp1, p = self.y.shape
        self.N, self.T, self.p = N, Tp1 - 1, p

        self.include_inputs = (self.u is not None) and (self.u.shape[-1] > 0)

        self._full_context = window_L is None
        if self._full_context:
            if self.T < self.H:
                raise ValueError(f"Need T >= H for full-context mode; got T={self.T}, H={self.H}.")
            self.L = self.T - self.H + 1
            t0 = self.L
            self._anchors = [(n, t0) for n in range(self.N)]
        else:
            self.L = int(window_L)
            if self.L < 1:
                raise ValueError("window_L must be >= 1 or None.")
            if self.T - self.H + 1 < self.L:
                raise ValueError(f"Sequence too short for L={self.L}, H={self.H}, T={self.T}.")
            self._anchors = [(n, t0) for n in range(self.N)
                             for t0 in range(1, self.T - self.H + 2)]

    def __len__(self):
        return len(self._anchors)

    def __getitem__(self, idx):
        n, t0 = self._anchors[idx]

        if self._full_context:
            y_hist = self.y[n, 0:t0, :]
            y_dense = build_dense_targets(self.y[n], H=self.H, start=0, L=t0)
            item = {
                "y": torch.from_numpy(y_hist),
                "y_target": torch.from_numpy(y_dense),
            }
            if self.include_inputs:
                u_hist = self.u[n, 0:t0, :]
                item["u"] = torch.from_numpy(u_hist)
            return item

        L = self.L
        k = min(L, t0)
        start = t0 - k
        y_ctx = self.y[n, start:t0, :]

        if k < L:
            pad = np.zeros((L - k, self.p), dtype=np.float32)
            y_hist = np.concatenate([pad, y_ctx], axis=0)
            mask = np.concatenate([np.zeros(L - k, dtype=np.int64),
                                   np.ones(k, dtype=np.int64)], 0)
        else:
            y_hist = y_ctx
            mask = np.ones(L, dtype=np.int64)

        y_next = self.y[n, t0:t0 + self.H, :]

        item = {
            "y": torch.from_numpy(y_hist),
            "y_target": torch.from_numpy(y_next),
            "attn_mask": torch.from_numpy(mask),
        }
        if self.include_inputs:
            m = self.m
            u_ctx = self.u[n, start:t0, :]
            if k < L:
                upad = np.zeros((L - k, m), dtype=np.float32)
                u_hist = np.concatenate([upad, u_ctx], axis=0)
            else:
                u_hist = u_ctx
            item["u"] = torch.from_numpy(u_hist)
        return item


def make_train_val_datasets(
    pkl_path,
    H,
    L,
    train_ratio,
    seed,
    normalize,
):
    """Build train/val datasets for sliding-window or full-context forecasting."""
    pkl_path = Path(pkl_path)
    data = load_pickle(pkl_path)

    pstr = str(pkl_path).lower()
    dataset_type = "lti" if "lti" in pstr else ("drone" if "drone" in pstr else "unknown")

    y = np.asarray(data["measurements"], dtype=np.float32)
    u = data.get("inputs", None)
    if u is not None:
        u = np.asarray(u, dtype=np.float32)
        m = u.shape[-1]

    N, _, p = y.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * N))
    idx_train = np.sort(idx[:n_train])
    idx_val = np.sort(idx[n_train:]) if n_train < N else np.array([], dtype=int)

    y_tr, y_va = y[idx_train], y[idx_val]
    if u is None:
        u_tr = u_va = None
    else:
        u_tr, u_va = u[idx_train], u[idx_val]

    if normalize:
        y_trn, u_trn, stats = normalize_y_u(y_tr, u_tr)
        y_van, u_van, _ = normalize_y_u(y_va, u_va, stats)
    else:
        stats = {
            "mu_y":  np.zeros(p, dtype=np.float32),
            "std_y": np.ones (p, dtype=np.float32),
            "mu_u":  (None if u is None else np.zeros(m, dtype=np.float32)),
            "std_u": (None if u is None else np.ones (m, dtype=np.float32)),
            "eps":   0.0,
        }
        y_trn, u_trn = y_tr, u_tr
        y_van, u_van = y_va, u_va

    train_ds = HorizonWindowDataset(
        measurements=y_trn,
        inputs=u_trn,
        window_L=L,
        horizon_H=H,
        dataset_type=dataset_type,
    )

    val_ds = None
    if len(idx_val) > 0:
        val_ds = HorizonWindowDataset(
            measurements=y_van,
            inputs=u_van,
            window_L=L,
            horizon_H=H,
            dataset_type=dataset_type,
        )

    train_ds.norm_stats = stats
    if val_ds is not None:
        val_ds.norm_stats = stats

    return train_ds, val_ds