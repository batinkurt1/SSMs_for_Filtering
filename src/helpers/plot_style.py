import matplotlib as mpl
from .cfg_loader import load_yaml

PLOT_YAML = load_yaml("configs/plot.yaml")


def set_plot_style():
    """Matplotlib style tuned for paper figures."""
    mpl.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.1,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        "axes.grid": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    })


def apply_plot_axis_style(ax):
    axis_cfg = dict(PLOT_YAML["axis"])
    ax.set_xlabel(axis_cfg["xlabel"])
    ax.set_ylabel(axis_cfg["ylabel"])
    ax.set_ylim(*tuple(axis_cfg["ylim"]))

    if bool(PLOT_YAML["use_log_y"]):
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.6)


def save_figure(fig, basepath, *, save_pdf, dpi=300):
    if bool(save_pdf):
        fig.savefig(basepath.with_suffix(".pdf"), dpi=int(dpi))
