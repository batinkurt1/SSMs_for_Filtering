from copy import deepcopy
from helpers.cfg_loader import load_yaml 
from training.train_model import train_model


def _build_variant_cfgs(cfg: dict):
    use_standard = bool(cfg["standard"])
    use_colored_noise = bool(cfg["colored_noise"])
    use_length_generalization = bool(cfg["length_generalization"])

    variant_cfgs = []

    if use_standard:
        cfg_standard = deepcopy(cfg)
        cfg_standard["standard"] = True
        cfg_standard["colored_noise"] = False
        cfg_standard["length_generalization"] = False
        variant_cfgs.append(("standard", cfg_standard))

    if use_colored_noise:
        cfg_colored = deepcopy(cfg)
        cfg_colored["standard"] = False
        cfg_colored["colored_noise"] = True
        cfg_colored["length_generalization"] = False
        variant_cfgs.append(("colored_noise", cfg_colored))

    if use_length_generalization:
        cfg_length = deepcopy(cfg)
        cfg_length["standard"] = False
        cfg_length["colored_noise"] = False
        cfg_length["length_generalization"] = True
        variant_cfgs.append(("length_generalization", cfg_length))

    if len(variant_cfgs) == 0:
        raise ValueError("No training variants selected. Set at least one of: standard, colored_noise, length_generalization.")

    return variant_cfgs


def train():
    cfg = load_yaml("configs/train.yaml")
    variant_cfgs = _build_variant_cfgs(cfg)

    for variant_name, variant_cfg in variant_cfgs:
        print(
            f"[train] variant={variant_name} | standard={bool(variant_cfg['standard'])} "
            f"colored_noise={bool(variant_cfg['colored_noise'])} "
            f"length_generalization={bool(variant_cfg['length_generalization'])}"
        )
        for name in variant_cfg["models"]:
            train_model(deepcopy(variant_cfg), name)

    print("Done.")