# plotting/plot_results.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from helpers.cfg_loader import load_yaml
from helpers.load_pickle import load_pickle
from helpers.plot_style import set_plot_style, apply_plot_axis_style, save_figure

PLOT_YAML = load_yaml("configs/plot.yaml")


def load_cases_from_paths(case_paths: dict[str, Path], case_order: list[str]):
    cases = {}
    meta = {"system": "", "H": None, "L": None, "n_traj": None}

    for case_name in case_order:
        path = case_paths[case_name]
        resolved = path
        if not resolved.exists():
            raise FileNotFoundError(f"Missing result file for case '{case_name}': {resolved}")

        data = load_pickle(resolved)
        case_obj = data

        cases[case_name] = case_obj
        if not meta["system"] and "system" in data:
            meta["system"] = data["system"]
        if meta["H"] is None and "H" in data:
            meta["H"] = data["H"]
        if meta["L"] is None and "L" in data:
            meta["L"] = data["L"]
        if meta["n_traj"] is None and "n_traj" in data:
            meta["n_traj"] = data["n_traj"]

    if len(cases) == 0:
        raise FileNotFoundError("No case result pickle found from CASE_RESULTS_PATHS.")

    return {"cases": cases, **meta}


def _extract_case_results(raw_out: dict):
    """
    Returns dict case_name -> {
        baseline:(A,H),
        baseline_label:str,
        baseline_std:(A,H)|None,
        models:[{"label":str, "rms":(A,H), "std":(A,H)|None}]
    } where
    all arrays are RMS over output dimensions.
    """
    def parse_case(case_obj):
        model_labels = dict(PLOT_YAML["model_labels"])
        baseline_labels = dict(PLOT_YAML["baseline_labels"])

        baseline_obj = case_obj["baseline"]
        baseline = np.asarray(baseline_obj["rms_time_h"])
        if baseline.ndim != 2:
            raise ValueError(f"Expected (A,H); got {baseline.shape}")
        baseline_label = baseline_labels[str(baseline_obj["kind"]).strip()]

        baseline_std = None
        if "rms_ah_by_traj" in baseline_obj:
            brms_traj = np.asarray(baseline_obj["rms_ah_by_traj"], dtype=float)
            baseline_std = brms_traj.std(axis=0)

        models = []
        for k, v in case_obj.items():
            if not k.startswith("model_"):
                continue
            name = v.get("name", k.replace("model_", ""))
            label = model_labels[str(name).strip()]
            arr = np.asarray(v["rms_time_h"])
            if arr.ndim != 2:
                raise ValueError(f"Expected (A,H); got {arr.shape}")
            std = None
            if "rms_ah_by_traj" in v:
                mrms_traj = np.asarray(v["rms_ah_by_traj"], dtype=float)
                std = mrms_traj.std(axis=0)
            models.append({"label": label, "rms": arr, "std": std})

        return {
            "baseline": baseline,
            "baseline_label": baseline_label,
            "baseline_std": baseline_std,
            "models": models,
        }

    out = {}
    for name, obj in raw_out["cases"].items():
        out[str(name)] = parse_case(obj)
    return out


def _plot_series(ax, t, y, label, ystd=None):
    color_map = dict(PLOT_YAML["color_map"])
    color = color_map[label]
    ax.plot(t, y, label=label, color=color)
    if bool(PLOT_YAML["rms_show_error_bands"]) and ystd is not None:
        lower = np.clip(y - ystd, 0.0, None)
        upper = y + ystd
        ax.fill_between(t, lower, upper, color=color, alpha=0.14, linewidth=0)


def _plot_case_on_axis(ax, case, case_name, h_idx):
    base_rms = case["baseline"][:, h_idx]
    t = np.arange(1, base_rms.shape[0] + 1)
    baseline_label = case["baseline_label"]

    bstd = None
    if case["baseline_std"] is not None:
        bstd = case["baseline_std"][:, h_idx]
    _plot_series(ax, t, base_rms, baseline_label, bstd)

    for model in case["models"]:
        label = model["label"]
        rms = model["rms"][:, h_idx]
        mstd = None
        if model["std"] is not None:
            mstd = model["std"][:, h_idx]
        _plot_series(ax, t, rms, label, mstd)

    if bool(PLOT_YAML["show_switch_marker"]) and case_name == "dynamics_switching":
        t_switch = len(t) // 2
        ax.axvline(t_switch, linestyle="--", linewidth=1.0, color="0.4", alpha=0.8)


def plot_rms_cases(raw_out: dict, outdir: Path, case_order: list[str], standalone_case_order: list[str], rms_horizon: int = 1, show: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)
    cases = _extract_case_results(raw_out)

    # Infer H from first available case
    first_case = next(iter(cases.values()))
    _, H = first_case["baseline"].shape
    h_idx = max(0, min(H - 1, int(rms_horizon) - 1))

    plot_cfg = dict(PLOT_YAML["plot_cfg"])
    nrows, ncols = tuple(plot_cfg["grid"])
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=tuple(PLOT_YAML["figsize_2x2"]),
        sharey=True,
        constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)

    for i, case_name in enumerate(case_order):
        if i >= len(axes):
            break
        ax = axes[i]
        case = cases[case_name]
        _plot_case_on_axis(ax, case, case_name, h_idx)

        apply_plot_axis_style(ax)
        if i % ncols != 0:
            ax.set_ylabel("")
        ax.legend(loc="upper right", frameon=False)

    for j in range(min(len(case_order), len(axes)), len(axes)):
        axes[j].axis("off")

    basepath = outdir / f"rms_cases_2x2_h{h_idx+1}"
    save_figure(fig, basepath, save_pdf=bool(plot_cfg["save_pdf"]), dpi=int(PLOT_YAML["rms_case_dpi"]))

    # Save each case separately too
    if bool(plot_cfg["save_individual_cases"]):
        for case_name in standalone_case_order:
            case = cases[case_name]

            fig_i, ax_i = plt.subplots(1, 1, figsize=tuple(PLOT_YAML["figsize_single"]), constrained_layout=True)
            _plot_case_on_axis(ax_i, case, case_name, h_idx)

            apply_plot_axis_style(ax_i)
            ax_i.legend(loc="upper right", frameon=False)

            basepath_i = outdir / f"rms_case_{case_name}_h{h_idx+1}"
            save_figure(fig_i, basepath_i, save_pdf=bool(plot_cfg["save_pdf"]), dpi=int(PLOT_YAML["rms_case_dpi"]))

            if show:
                plt.show()
            else:
                plt.close(fig_i)

            print(f"Saved: {basepath_i.with_suffix('.pdf')}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved: {basepath.with_suffix('.pdf')}")


def plot():
    set_plot_style()
    case_order = list(PLOT_YAML["case_order"])
    standalone_case_order = list(PLOT_YAML["standalone_case_order"])
    case_results_paths = {
        str(k): Path(v) for k, v in PLOT_YAML["case_results_paths"].items()
    }
    raw_for_plot = load_cases_from_paths(case_results_paths, case_order=case_order)

    plot_rms_cases(
        raw_for_plot,
        outdir=Path(PLOT_YAML["outdir"]),
        case_order=case_order,
        standalone_case_order=standalone_case_order,
        rms_horizon=int(PLOT_YAML["rms_horizon"]),
        show=bool(PLOT_YAML["show_figs"]),
    )


if __name__ == "__main__":
    plot()