"""Reusable figure components.

Notebooks compose animations out of these rather than hand-rolling axes, so a
feature map drawn in notebook 01 looks identical to one drawn in notebook 07.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from cnnviz import style, text

__all__ = [
    "draw_matrix",
    "highlight_window",
    "draw_kernel",
    "draw_feature_maps",
    "frame_header",
    "progress_bar",
    "caption",
]


# --------------------------------------------------------------------------
# Animation furniture
#
# Every GIF in the series wears the same frame: a title block top-left, an
# optional caption under it, and a slim progress rule along the bottom. Keeping
# these in one place is what makes eight separately-authored animations read as
# one piece of work.
# --------------------------------------------------------------------------

def frame_header(fig, title: str, subtitle: str | None = None) -> None:
    """Draw the standard title block in the figure's top-left corner.

    Positioned in figure coordinates rather than as an axis title so it stays
    put regardless of how the panels below are laid out — a title that drifts
    between frames is the fastest way to make an animation look unfinished.
    """
    fig.text(
        0.012, 0.965, title, ha="left", va="top",
        fontsize=14, color=style.INK_PRIMARY,
    )
    if subtitle:
        fig.text(
            0.012, 0.895, subtitle, ha="left", va="top",
            fontsize=10.5, color=style.INK_SECONDARY,
        )


def caption(fig, label: str, y: float = 0.055, x: float = 0.012,
            fontsize: float = 10, ha: str = "left") -> None:
    """Draw an explanatory line along the bottom of the frame.

    This is where the animation says what the viewer should be noticing right
    now. It changes between frames; the header does not.
    """
    fig.text(
        x, y, label, ha=ha, va="bottom",
        fontsize=fontsize, color=style.INK_SECONDARY,
    )


def progress_bar(fig, fraction: float, y: float = 0.022,
                 x: float = 0.012, width: float = 0.976,
                 height: float = 0.006) -> None:
    """Draw a slim progress rule across the bottom of the frame.

    A GIF has no scrubber, so without this the viewer cannot tell whether they
    are watching the start of a long animation or the end of a short one, nor
    where the loop restarts.
    """
    fraction = min(max(fraction, 0.0), 1.0)

    fig.add_artist(Rectangle(
        (x, y), width, height,
        transform=fig.transFigure, facecolor=style.GRID,
        edgecolor="none", zorder=5,
    ))
    fig.add_artist(Rectangle(
        (x, y), width * fraction, height,
        transform=fig.transFigure, facecolor=style.CATEGORICAL[0],
        edgecolor="none", zorder=6,
    ))


def _format_cell(value: float) -> str:
    """Compact label for an annotated cell, in the active language.

    Integral values lose their decimals ("2" not "2,00") because the kernel of
    a hand-built edge detector is easier to read as integers; everything else
    keeps two places and uses the language's decimal separator.
    """
    if value == int(value):
        return text.MINUS + str(abs(int(value))) if value < 0 else str(int(value))
    return text.num(value, 2)


def draw_matrix(
    ax: Axes,
    values: np.ndarray,
    *,
    mode: str = "magnitude",
    norm=None,
    annotate: bool = False,
    title: str | None = None,
    fontsize: int = 8,
    title_fontsize: float = 11,
    grid: bool = True,
):
    """Render a 2-D array as a cell grid.

    Args:
        ax: Target axes.
        values: 2-D array to draw. Pass a ``numpy.ma.MaskedArray`` to leave
            cells blank — masked entries render in :data:`style.EMPTY`, which
            reads as "no value yet" and stays distinct from a genuine zero.
        mode: Which encoding to use.

            * ``"magnitude"`` — unsigned quantities (activations). Sequential
              blue, anchored at zero.
            * ``"signed"`` — anything that can be negative (weights,
              gradients, pre-activations). Diverging, centred on zero. Using
              ``"magnitude"`` here renders ``-0.5`` and ``0`` almost
              identically and hides the sign structure entirely.
            * ``"pixels"`` — raw input imagery. Greyscale, because a digit is
              a photograph rather than an encoding, and colouring it makes it
              compete with the encoded panels beside it.
        norm: Explicit colour scale, overriding the per-call one derived from
            ``values``. **Pass this whenever the panel is one frame of an
            animation.** The default scale is computed from the data in front
            of it, so a panel holding tiny values renders them fully saturated
            — across frames that makes a weak response look identical to a
            strong one, which inverts the story the animation is telling.
            Build a fixed scale from the whole sequence with
            :func:`style.signed_norm` or :func:`style.magnitude_norm`.
        annotate: Print the numeric value inside each cell. Only legible up
            to roughly 8x8; beyond that the text collides.
        title: Optional axis title.
        fontsize: Size of the annotation text.
        title_fontsize: Size of the title. Worth lowering in a tight layout:
            titles are left-aligned from the panel's edge, and a translated
            label can be much longer than its English original — "Feature map"
            becomes "Mapa de características" — so a title that fitted in one
            language will overrun the canvas in another.
        grid: Draw hairlines between cells so individual pixels stay countable.

    Returns:
        The ``AxesImage`` created, so callers can animate ``set_data`` on it.

    Raises:
        ValueError: If ``mode`` is not one of the three encodings.
    """
    values = np.asarray(values) if not np.ma.isMaskedArray(values) else values

    if mode == "signed":
        norm = norm or style.signed_norm(np.ma.filled(values, 0.0))
        cmap = style.CMAP_SIGNED
    elif mode == "magnitude":
        norm = norm or style.magnitude_norm(np.ma.filled(values, 0.0))
        cmap = style.CMAP_MAGNITUDE
    elif mode == "pixels":
        norm = norm or style.magnitude_norm(np.ma.filled(values, 0.0))
        cmap = plt.get_cmap(style.CMAP_PIXELS)
    else:
        raise ValueError(
            f"mode must be 'magnitude', 'signed' or 'pixels'; got {mode!r}"
        )

    cmap = cmap.with_extremes(bad=style.EMPTY)

    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
    style.bare(ax)

    if grid and max(values.shape) <= 32:
        ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
        # GRID, not SURFACE: a surface-coloured hairline is invisible against a
        # white cell, which makes a zero-valued pixel row look like it is not
        # part of the matrix at all.
        ax.grid(which="minor", color=style.GRID, linewidth=0.8)
        ax.tick_params(which="minor", length=0)

    if annotate:
        # Choose ink per cell by luminance of its fill, so text stays readable
        # against both the palest and the darkest end of the ramp.
        rgba = cmap(norm(np.ma.filled(values, 0.0)))
        luminance = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        mask = np.ma.getmaskarray(values)
        for (r, c), value in np.ndenumerate(np.ma.filled(values, 0.0)):
            if mask[r, c]:
                continue
            ax.text(
                c, r, _format_cell(float(value)),
                ha="center", va="center", fontsize=fontsize,
                color="#ffffff" if luminance[r, c] < 0.55 else style.INK_PRIMARY,
            )

    if title:
        ax.set_title(title, fontsize=title_fontsize, color=style.INK_PRIMARY)

    return image


def highlight_window(
    ax: Axes,
    row: int,
    col: int,
    size: int,
    color: str | None = None,
    linewidth: float = 2.0,
):
    """Outline the receptive field currently being read.

    Drawn as a stroked rectangle with no fill so the underlying values stay
    visible — the viewer needs to see *what* is under the window, not just
    where it is. A surface-coloured underlay separates the stroke from
    whatever sits beneath it, so the marker survives on both ends of every
    ramp in either theme.

    Args:
        color: Stroke colour. Defaults to the active theme's focus ink —
            resolved on each call, not at import, so it follows a theme
            switched after this module was loaded.
    """
    color = color or style.FOCUS
    for width, edge, z in (
        (linewidth + 2.0, style.SURFACE, 9),
        (linewidth, color, 10),
    ):
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5), size, size,
                fill=False, edgecolor=edge, linewidth=width, zorder=z,
            )
        )


def draw_kernel(ax: Axes, kernel: np.ndarray, title: str = "Kernel"):
    """Draw a single kernel with its weights printed — always signed."""
    return draw_matrix(
        ax, kernel, mode="signed", annotate=True, title=title, fontsize=9
    )


def draw_feature_maps(
    fig,
    maps: np.ndarray,
    *,
    ncols: int = 8,
    mode: str = "magnitude",
    titles: list[str] | None = None,
    suptitle: str | None = None,
    shared_scale: bool = True,
):
    """Lay out a stack of feature maps as a contact sheet.

    Args:
        fig: Target figure; its existing axes are not cleared.
        maps: Array of shape ``(C, H, W)``.
        ncols: Panels per row.
        mode: Encoding, as in :func:`draw_matrix`.
        titles: Per-panel labels.
        suptitle: Figure-level heading.
        shared_scale: Normalise every panel against the *same* range. Keep
            this on for comparisons across channels — per-panel scaling makes
            a near-dead channel look as active as a strongly firing one, which
            is the single most misleading thing a feature-map grid can do.

    Returns:
        The list of images created.
    """
    maps = np.asarray(maps)
    n = maps.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    signed = mode == "signed"
    norm_for = style.signed_norm if signed else style.magnitude_norm
    cmap = {
        "signed": style.CMAP_SIGNED,
        "magnitude": style.CMAP_MAGNITUDE,
    }.get(mode) or plt.get_cmap(style.CMAP_PIXELS)

    shared = norm_for(maps) if shared_scale else None

    axes = fig.subplots(nrows, ncols, squeeze=False)
    images = []

    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.set_visible(False)
            continue

        panel_norm = shared if shared is not None else norm_for(maps[k])
        images.append(
            ax.imshow(maps[k], cmap=cmap, norm=panel_norm, interpolation="nearest")
        )
        style.bare(ax)
        if titles and k < len(titles):
            ax.set_title(titles[k], fontsize=9, color=style.INK_SECONDARY)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=style.INK_PRIMARY, x=0.02, ha="left")

    return images
