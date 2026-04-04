import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from datasources.normalization import denorm_y

class BaseForecastModel(pl.LightningModule):
    def __init__(self, backbone, H,
                 lr, weight_decay,
                 loss_reduction, norm_stats):
        super().__init__()
        self.backbone = backbone
        self.H = int(H)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_reduction = loss_reduction
        self.norm_stats = norm_stats

    def reduce_loss_vec(self, loss_vec, reduction):
        if reduction == "mean":
            return loss_vec.mean()
        elif reduction == "sum":
            return loss_vec.sum()
        else:  # "none"
            return loss_vec  # shape (B,)

    def forward(self, batch, batch_idx=None):
        # Inputs
        y = batch["y"]                       # (B, L, p) or (B, L_full, p)
        u = batch.get("u", None)             # (B, L, m) or None
        attn_mask = batch.get("attn_mask", None)  # (B, L) with 1 real, 0 pad in sliding mode

        pred = self.backbone(y, u, attn_mask=attn_mask)

        target = batch["y_target"]       # (B, H, p) or (B, L, H, p)

        if target.ndim == 3:
            # ----- Sliding-window mode: target (B, H, p) -----
            pred_last = self.ensure_last_row(pred)  # (B, H, p)
            out = self.loss_metrics_sliding(pred_last, target)

        elif target.ndim == 4:
            # ----- Full-context dense mode: target (B, L, H, p) -----
            if pred.ndim != 4:
                raise ValueError(f"Dense mode expects pred (B,L,H,p), got {pred.shape}")
            out = self.loss_metrics_dense(pred, target)

        else:
            raise ValueError(f"Unexpected target ndim={target.ndim}; expected 3 or 4.")

        return out

    def ensure_last_row(self, pred):
        """
        Sliding mode accepts either:
          - (B, L, H, p): take last row
          - (B, H, p):    already last-row
        """
        if pred.ndim == 4:
            return pred[:, -1]  # (B, H, p)
        elif pred.ndim == 3:
            return pred
        else:
            raise ValueError(f"Sliding mode expects pred (B,L,H,p) or (B,H,p), got {pred.shape}")

    def calculate_metrics(self, pred, target, mode, suffix=""):
        """
        mode: 'sliding' (pred,target: B,H,p)  or  'dense' (pred,target: B,L,H,p)
        Fills a dict with loss/metrics (normalized space); caller can pass suffix="_denorm".
        """
        out = {}

        if mode == "sliding":
            if pred.shape != target.shape or pred.ndim != 3:
                raise ValueError(f"Expected (B,H,p) for sliding, got {pred.shape}, {target.shape}")
            # (B,H)
            per_mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
            # (B,)
            loss_vec = per_mse.mean(dim=1)
            loss = self.reduce_loss_vec(loss_vec, self.loss_reduction)

            # last horizon
            pred_last = pred[:, -1, :]   # (B,p)
            tgt_last  = target[:, -1, :] # (B,p)
            mae  = (pred_last - tgt_last).abs().mean()
            rmse = torch.sqrt(F.mse_loss(pred_last, tgt_last))

            # per-horizon summary (H,)
            step_mse = per_mse.mean(dim=0)

        elif mode == "dense":
            if pred.shape != target.shape or pred.ndim != 4:
                raise ValueError(f"Expected (B,L,H,p) for dense, got {pred.shape}, {target.shape}")
            # (B,L,H)
            per_mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
            # (B,)
            loss_vec = per_mse.mean(dim=(1,2))
            loss = self.reduce_loss_vec(loss_vec, self.loss_reduction)

            # last token, last horizon
            pred_last = pred[:, -1, -1, :]  # (B,p)
            tgt_last  = target[:, -1, -1, :]# (B,p)
            mae  = (pred_last - tgt_last).abs().mean()
            rmse = torch.sqrt(F.mse_loss(pred_last, tgt_last))

            # per-horizon summary (H,)
            step_mse = per_mse.mean(dim=(0,1))

        out.update({
            f"optimized_loss{suffix}": loss if suffix == "" else loss.detach(),
            f"loss_mse{suffix}":       loss.detach(),
            f"metric_mae{suffix}":     mae.detach(),
            f"metric_rmse{suffix}":    rmse.detach(),
            f"metric_mse_last{suffix}": step_mse[-1].detach(),
        })
        for h, mseh in enumerate(step_mse):
            out[f"metric_mse_h{h}{suffix}"] = mseh.detach()

        return out
    
    def loss_metrics_sliding(self, pred_last, target):
        if pred_last.shape != target.shape or pred_last.ndim != 3:
            raise ValueError(f"Expected pred_last,target (B,H,p). Got {pred_last.shape}, {target.shape}")
        out = self.calculate_metrics(pred_last, target, mode="sliding", suffix="")
        with torch.no_grad():
            out.update(self.calculate_metrics(
                denorm_y(pred_last, self.norm_stats),
                denorm_y(target, self.norm_stats),
                mode="sliding", suffix="_denorm"))
        return out

    def loss_metrics_dense(self, pred, target):
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError(f"Expected pred,target (B,L,H,p). Got {pred.shape}, {target.shape}")
        out = self.calculate_metrics(pred, target, mode="dense", suffix="")
        with torch.no_grad():
            out.update(self.calculate_metrics(
                denorm_y(pred, self.norm_stats),
                denorm_y(target, self.norm_stats),
                mode="dense", suffix="_denorm"))
        return out

    def log_output(self, output_dict, phase):
        on_step = (phase == "train")          # only train logs each step
        for k, v in output_dict.items():
            if "loss" in k or "metric" in k:
                if isinstance(v, torch.Tensor):
                    if v.ndim > 0:
                        v = v.mean()
                    v = v.detach()
                self.log(
                    f"{phase}_{k}",
                    v,
                    on_step=on_step,
                    on_epoch=True,           
                    prog_bar=("loss" in k),
                    logger=True,
                )

    def training_step(self, batch, batch_idx):
        out = self(batch, batch_idx=batch_idx)
        self.log_output(out, "train")
        return {"loss": out["optimized_loss"], "output_dict": out}

    def validation_step(self, batch, batch_idx):
        out = self(batch, batch_idx=batch_idx)
        self.log_output(out, "val")
        return {"loss": out["optimized_loss"], "output_dict": out}

    def test_step(self, batch, batch_idx):
        out = self(batch, batch_idx=batch_idx)
        self.log_output(out, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
