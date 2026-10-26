"""Small plotting helpers with one consistent style for all studies."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical colours, always assigned in this order (never cycled)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
# One colour per target, used in every figure
TARGET_COLORS = dict(zip(("FC", "FS", "FS_S", "FS_A"), COLORS))
GRAY = "#898781"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32525"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": False, "grid.color": "#e1e0d9",
    "legend.frameon": False, "lines.linewidth": 2,
})


def color_for(items):
    """Fixed colour per item, in order of first appearance."""
    return {item: COLORS[i % len(COLORS)] for i, item in enumerate(items)}


def show_matrix(ax, M, title="", vmax=None, diverging=True, colorbar=True):
    """Matrix image with the diagonal masked; diverging scales are centred
    on 0."""
    M = np.array(M, dtype=float)
    if M.ndim == 2 and M.shape[0] == M.shape[1]:
        np.fill_diagonal(M, np.nan)
    if vmax is None:
        vmax = np.nanmax(np.abs(M)) if diverging else np.nanmax(M)
    kw = (dict(cmap=DIVERGING, vmin=-vmax, vmax=vmax) if diverging
          else dict(cmap=SEQUENTIAL, vmin=np.nanmin(M), vmax=vmax))
    im = ax.imshow(M, interpolation="nearest", **kw)
    ax.set(title=title, xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def save(fig, path):
    fig.savefig(path)
    print(f"saved {path}")
