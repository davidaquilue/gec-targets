"""Small plotting helpers with one consistent style for all studies."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (LinearSegmentedColormap, LogNorm, Normalize,
                               SymLogNorm)
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Categorical colours, always assigned in this order (never cycled)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
# One colour per target, used in every figure
TARGET_COLORS = dict(zip(("FC", "FS", "FS_S", "FS_A"), COLORS))
GRAY = "#898781"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a32525"])
# The two halves of DIVERGING for signed values of one sign, so that red is
# always positive and blue negative, darker = larger in magnitude; they start
# from a light tint, not the gray of 0, which is outside their range
POSITIVE = LinearSegmentedColormap.from_list(
    "reds", ["#f8dcdc", "#f0a3a3", "#e66767", "#c44646", "#a32525"])
NEGATIVE = LinearSegmentedColormap.from_list(
    "blues_r", ["#104281", "#2563b3", "#3987e5", "#8cb9f0", "#dbe8fa"])
# Unsigned quantities (counts, power): neutral grays, never read as a sign
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "grays", ["#e8e7e2", "#c3c2b7", "#898781", "#52514e", "#262523"])

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


def corr_norm(values):
    """Scale of a panel of signed values (e.g. correlations). Diverging,
    centred on 0, when the panel has both signs (FS_A can be fitted with the
    wrong sign); else over the panel's own range, so that it keeps its
    contrast, in the half of the diverging map of its sign. Positive is
    always red and negative blue, darker = larger in magnitude."""
    lo, hi = np.nanmin(values, initial=np.inf), np.nanmax(values, initial=-np.inf)
    if not np.isfinite([lo, hi]).all():
        return Normalize(-1, 1), DIVERGING
    if lo < 0 < hi:
        vmax = max(-lo, hi)
        return Normalize(-vmax, vmax), DIVERGING
    return Normalize(lo, hi), NEGATIVE if hi <= 0 else POSITIVE


def heat_panel(ax, grid, title, norm, cmap, fmt, best=False, outlines=()):
    """grid: DataFrame of a hyperparameter grid (rows x columns, named, e.g.
    by DataFrame.pivot), log-spaced values drawn as equal cells, each
    annotated.

    best: outline the largest cell in black. outlines: cells to mark, dicts
    with "cell" (row value, column value), "color" and "linestyle"; a cell
    that is not on the grid is skipped. Marks are drawn inside their cell
    (two neighbours stay apart), several on one cell further inset."""
    Z = np.ma.masked_invalid(grid.to_numpy(float))
    im = ax.pcolormesh(Z, cmap=cmap, norm=norm, edgecolors="#fcfcfb",
                       linewidth=1.5)
    for (i, j), v in np.ndenumerate(Z.filled(np.nan)):
        if np.isfinite(v):
            r, g, b = np.ravel(cmap(norm(v)))[:3]
            ink = "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 \
                else "#0b0b0b"
            text = fmt(v) if callable(fmt) else format(v, fmt)
            ax.text(j + 0.5, i + 0.5, text, ha="center",
                    va="center", fontsize=6, color=ink)
    marks = []
    if best and Z.count():
        marks.append((*np.unravel_index(np.ma.argmax(Z), Z.shape),
                      "#0b0b0b", "-"))
    for o in outlines:
        i = np.flatnonzero(np.isclose(grid.index.to_numpy(float), o["cell"][0]))
        j = np.flatnonzero(np.isclose(grid.columns.to_numpy(float),
                                      o["cell"][1]))
        if len(i) and len(j):
            marks.append((i[0], j[0], o["color"], o.get("linestyle", "-")))
    drawn = {}
    for i, j, color, linestyle in marks:
        d = 0.05 + 0.1 * drawn.get((i, j), 0)
        drawn[i, j] = drawn.get((i, j), 0) + 1
        ax.add_patch(Rectangle((j + d, i + d), 1 - 2 * d, 1 - 2 * d,
                               fill=False, edgecolor=color,
                               linestyle=linestyle, linewidth=1.5))
    ax.set(title=title, xlabel=grid.columns.name, ylabel=grid.index.name,
           xticks=np.arange(grid.shape[1]) + 0.5,
           yticks=np.arange(grid.shape[0]) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in grid.columns], rotation=90)
    ax.set_yticklabels([f"{v:g}" for v in grid.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if isinstance(norm, LogNorm):
        ticks = [v for v in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                 if norm.vmin <= v <= norm.vmax] or [norm.vmin, norm.vmax]
        cbar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cbar.minorticks_off()
    elif isinstance(norm, SymLogNorm):
        top = int(np.floor(np.log10(norm.vmax)))
        ticks = sorted({0} | {s * 10 ** k for k in range(top + 1)
                              for s in (-1, 1)})
        cbar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        cbar.minorticks_off()


def outline_legend(fig, styles):
    """Legend of the outlines above the figure; styles: dicts with "color",
    "linestyle", "label". Call after tight_layout."""
    handles = [Line2D([], [], color=s["color"], linestyle=s["linestyle"],
                      linewidth=1.5, label=s["label"]) for s in styles]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0), fontsize=8)


def save(fig, path):
    fig.savefig(path)
    print(f"saved {path}")
