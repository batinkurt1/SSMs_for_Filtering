import pickle
from pathlib import Path

import torch
import yaml

from sequence_models.GPT2Backbone import GPT2Backbone
from sequence_models.SelectiveSSMBackbone import SelectiveSSMBackbone
from sequence_models.BaseForecastModel import BaseForecastModel


def find_checkpoint(run_dir):
    ckpt_dir = Path(run_dir) / "checkpoints"
    best = ckpt_dir / "best.ckpt"
    if best.exists():
        return best
    cands = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

def load_model(run_dir, device="cpu"):
    """
    Build a BaseForecastModel from a run directory containing:
      - model_meta.yaml
      - norm_stats.pkl
      - checkpoints/best.ckpt (or newest *.ckpt)
    Returns: (model, meta)
    """
    run_dir = Path(run_dir)

    meta_path = run_dir / "model_meta.yaml"
    stats_path = run_dir / "norm_stats.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    typ = meta["type"]          # "gpt2" or "ssm"
    p   = int(meta["p"])
    m   = int(meta["m"])
    H   = int(meta["H"])
    arch= dict(meta.get("arch", {}))  # hyperparams for backbone

    # Optional training hyperparams (fallbacks kept for older metas)
    train_meta = dict(meta.get("train", {}))
    lr = float(train_meta.get("lr", 1e-3))
    weight_decay = float(train_meta.get("weight_decay", 0.01))
    loss_reduction = train_meta.get("loss_reduction", "mean")

    if typ == "gpt2":
        backbone = GPT2Backbone(p=p, m=m, H=H, **arch)
    elif typ == "ssm":
        backbone = SelectiveSSMBackbone(p=p, m=m, H=H, **arch)
    else:
        raise ValueError(f"Unknown model type: {typ}")

    model = BaseForecastModel(
        backbone=backbone, H=H,
        lr=lr, weight_decay=weight_decay,
        loss_reduction=loss_reduction,
        norm_stats=norm_stats
    ).to(device)

    ckpt_path = find_checkpoint(run_dir)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=True)
    return model, meta
