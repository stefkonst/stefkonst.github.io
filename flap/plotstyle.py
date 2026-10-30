"""Figure style of Konstantinidi et al. (Smart Mater. Struct. 2023; Compos. Sci. Technol. 2024): Times serif,
full axis box, light major + minor grid, labels 'Quantity symbol in unit', panel letters 'a)', MATLAB colour order,
voltage off blue / on red."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATLAB = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
OFF, ON = "#0000FF", "#FF0000"


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8, "axes.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=MATLAB), "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "axes.spines.top": True, "axes.spines.right": True, "legend.frameon": True, "legend.edgecolor": "0.3",
        "legend.fancybox": False, "figure.dpi": 150, "savefig.dpi": 150})


def finish(axs, letters=True):
    """Grid (major + minor) and panel letters a), b), ..."""
    axs = list(axs.flat) if hasattr(axs, "flat") else (list(axs) if isinstance(axs, (list, tuple)) else [axs])
    for k, a in enumerate(axs):
        a.minorticks_on()
        a.grid(True, which="major", color="0.82", lw=0.6)
        a.grid(True, which="minor", color="0.92", lw=0.4)
        a.set_axisbelow(True)
        if letters and len(axs) > 1:
            a.text(-0.17, 1.03, f"{chr(97 + k)})", transform=a.transAxes, fontsize=13, va="bottom")
