from torch.utils.data import DataLoader

from datasources.make_train_val_datasets import make_train_val_datasets

def build_loaders(cfg):
    """
    Returns:
      train_ds, val_ds, train_loader, val_loader, p, m
    """
    train_ds, val_ds = make_train_val_datasets(
        pkl_path=cfg["data_path"],
        H=cfg["H"],
        L=cfg["L"],
        train_ratio=cfg["train_ratio"],
        seed=cfg["seed"],
        normalize=cfg["normalize"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=False,
        persistent_workers=(cfg["num_workers"] > 0),
    )

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=False,
            persistent_workers=(cfg["num_workers"] > 0),
        )

    p, m = train_ds.p, train_ds.m

    return train_ds, val_ds, train_loader, val_loader, p, m
