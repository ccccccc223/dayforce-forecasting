"""Shared matplotlib style for report figures (light surface, validated 3-slot palette).

Clients keep fixed colours in every chart. Aqua is below 3:1 contrast on the surface,
so charts label series directly as well as using a legend.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLIENT_COLORS = {"longos": "#2a78d6", "moto": "#eb6834", "costa": "#1baf7a"}
CLIENT_LABELS = {"longos": "Longo's", "moto": "Moto", "costa": "Costa"}
CLIENTS = ["longos", "moto", "costa"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def label_end(ax, x, y, client, dx=0):
    """Direct label at a line's end, in secondary ink beside the coloured line."""
    ax.annotate(CLIENT_LABELS[client], (x, y), xytext=(4 + dx, 0), textcoords="offset points",
                va="center", color=INK_2, fontsize=9)
