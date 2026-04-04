import pickle
from pathlib import Path
import pytorch_lightning as pl
import yaml


from datasources.build_loaders import build_loaders
from training.build_trainer import build_trainer

from sequence_models.BaseForecastModel import BaseForecastModel
from sequence_models.GPT2Backbone import GPT2Backbone
from sequence_models.SelectiveSSMBackbone import SelectiveSSMBackbone
from sequence_models.MambaBackbone import MambaBackbone


def train_model(cfg, model_name):
    cfg = dict(cfg)
    colored_noise = bool(cfg["colored_noise"])
    length_generalization = bool(cfg["length_generalization"])

    if colored_noise:
        if not cfg["data_path_colored_noise"]:
            raise ValueError("colored_noise=True requires 'data_path_colored_noise' in config.")
        cfg["data_path"] = cfg["data_path_colored_noise"]
    elif length_generalization:
        if not cfg["data_path_length_generalization"]:
            raise ValueError("length_generalization=True requires 'data_path_length_generalization' in config.")
        cfg["data_path"] = cfg["data_path_length_generalization"]
    else:
        if not cfg["data_path_standard"]:
            raise ValueError("Standard case requires 'data_path_standard' in config.")
        cfg["data_path"] = cfg["data_path_standard"]

    train_ds, val_ds, train_loader, val_loader, p, m = build_loaders(cfg)
    print(f"[{model_name}] p={p}, m={m}, H={cfg['H']}, L={cfg['L']}, "
          f"train_batches={len(train_loader)}, val_batches={len(val_loader) if val_loader else 0}")

    if model_name == "gpt2":
        gpt_n_layer = int(cfg["gpt_n_layer"])
        gpt_d_model = int(cfg["gpt_d_model"])
        backbone = GPT2Backbone(
            p=p, m=m, H=cfg["H"],
            d_model=gpt_d_model, n_layer=gpt_n_layer, n_head=cfg["n_head"],
            dropout=cfg["dropout"], max_len=cfg["max_len"],
        )
        arch_type = "gpt2"
        arch_cfg = {
            "d_model": gpt_d_model,
            "n_layer": gpt_n_layer,
            "n_head":  cfg["n_head"],
            "dropout": cfg["dropout"],
            "max_len": cfg["max_len"],
        }
    elif model_name == "ssm":
        ssm_n_layer = int(cfg["ssm_n_layer"])
        ssm_d_model = int(cfg["ssm_d_model"])
        backbone = SelectiveSSMBackbone(
            p=p, m=m, H=cfg["H"],
            d_model=ssm_d_model, n_layer=ssm_n_layer, n_x=cfg["n_x"],
            s_A=cfg["s_A"], use_delta=cfg["use_delta"], fix_sA=cfg["fix_sA"],
            dropout=cfg["dropout"],
        )
        arch_type = "ssm"
        arch_cfg = {
            "d_model":  ssm_d_model,
            "n_layer":  ssm_n_layer,
            "n_x":      cfg["n_x"],
            "s_A":      cfg["s_A"],
            "use_delta":cfg["use_delta"],
            "fix_sA":   cfg["fix_sA"],
            "dropout":  cfg["dropout"],
        }
    elif model_name == "mamba":
        mamba_d_model = int(cfg["mamba_d_model"])
        mamba_n_layer = int(cfg["mamba_n_layer"])
        backbone = MambaBackbone(
            p=p, m=m, H=cfg["H"],
            d_model=mamba_d_model, n_layer=mamba_n_layer,
            d_state=cfg["d_state"], d_conv=cfg["d_conv"], expand=cfg["expand"],
            dropout=cfg["dropout"],
        )
        arch_type = "mamba"
        arch_cfg = {
            "d_model": mamba_d_model,
            "n_layer": mamba_n_layer,
            "d_state": cfg["d_state"],
            "d_conv":  cfg["d_conv"],
            "expand":  cfg["expand"],
            "dropout": cfg["dropout"],
        }
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: gpt2, ssm, mamba.")
    
    lm = BaseForecastModel(
        backbone=backbone, H=cfg["H"],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        loss_reduction=cfg["loss_reduction"],
        norm_stats=train_ds.norm_stats
    )

    u_tag = "input" if train_ds.m > 0 else "no_input"
    L_tag = f"L{cfg['L']}" if cfg["L"] is not None else "Lfull"
    epochs_tag = f"ep{cfg['epochs']}"
    case_tag = "-colored_noise" if colored_noise else ("-length_generalization" if length_generalization else "")
    run_dir = Path(cfg["outputs_dir"]) / train_ds.dataset_type / f"{model_name}-{u_tag}-H{cfg['H']}-S{train_ds.T}-{L_tag}-{epochs_tag}{case_tag}"

    cfg["model_name"] = model_name
    trainer, ckpt = build_trainer(run_dir, cfg, val_loader=val_loader)

    pl.seed_everything(42, workers=True)
    trainer.fit(lm, train_loader, val_loader)
    print(f"[{model_name}] Best checkpoint: {ckpt.best_model_path}")

    stats = train_ds.norm_stats
    if stats is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "norm_stats.pkl", "wb") as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{model_name}] Saved norm stats → {run_dir / 'norm_stats.pkl'}")
    meta = {
        "type": arch_type,                 # "gpt2" or "ssm"
        "p": int(p),
        "m": int(m),
        "H": int(cfg["H"]),
        "arch": arch_cfg,                  # hyperparameters needed to rebuild the backbone
        "train": {                        
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
            "loss_reduction": str(cfg["loss_reduction"]),
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "L": (None if cfg["L"] is None else int(cfg["L"])),
            "seed": int(cfg["seed"]),
            "colored_noise": bool(colored_noise),
            "length_generalization": bool(length_generalization),
            "data_path": str(cfg["data_path"]),
        },
        "data": {
            "dataset_type": train_ds.dataset_type  # "lti" or "drone"
        }
    }

    meta_path = run_dir / "model_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[{model_name}] Saved model meta to {meta_path}")