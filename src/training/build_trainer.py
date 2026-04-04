from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

def build_trainer(out_dir: Path, cfg, val_loader=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = "val_loss_mse" if val_loader is not None else "train_loss_mse"
    fname = f"{cfg['model_name']}-H{cfg['H']}-L{cfg['L']}-{{epoch:02d}}-{{{monitor}:.4f}}"
    ckpt = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename=fname,
        monitor=monitor, mode="min",
        save_top_k=1, save_last=True,
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    tb = TensorBoardLogger(save_dir=str(out_dir))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=cfg["epochs"],
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=cfg["grad_clip"],
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        callbacks=[ckpt, lrmon],
        logger=tb,
    )
    return trainer, ckpt
