"""Reusable figure components.

Notebooks compose animations out of these rather than hand-rolling axes, so a
feature map drawn in notebook 01 looks identical to one drawn in notebook 07.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle

from cnnviz import style, text

__all__ = [
    "draw_matrix",
    "highlight_window",
    "draw_kernel",
    "draw_maps",
    "draw_feature_maps",
    "frame_header",
    "progress_bar",
    "caption",
    "glyph_between",
    "arrow_between",
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


def draw_maps(
    axes,
    maps: np.ndarray,
    *,
    mode: str = "magnitude",
    norm=None,
    titles: list[str] | None = None,
    title_fontsize: float = 9,
    shared_scale: bool = True,
):
    """Paint a stack of feature maps into axes that already exist.

    The counterpart to :func:`draw_feature_maps`, which creates its own grid.
    **This is the one to use inside an animation**: a frame callback has to
    place its panels itself, and any helper that calls ``fig.subplots`` both
    ignores the frame's layout and re-solves the geometry each time it is
    called, which shows up as panels drifting by a pixel between frames.

    Args:
        axes: Flat sequence of axes, one per map. Extra axes are hidden, so a
            fixed grid can hold a stage with fewer channels than its
            neighbours without the layout moving.
        maps: Array of shape ``(C, H, W)``.
        mode: Encoding, as in :func:`draw_matrix`.
        norm: Explicit colour scale. Pass one built from the whole animation
            when these panels are a frame; see :func:`draw_matrix`.
        titles: Per-panel labels.
        title_fontsize: Size of those labels.
        shared_scale: Normalise every panel against the *same* range. Keep
            this on for comparisons across channels — per-panel scaling makes
            a near-dead channel look as active as a strongly firing one, which
            is the single most misleading thing a feature-map grid can do.

    Returns:
        The list of images created.
    """
    maps = np.asarray(maps)
    axes = list(axes)
    n = maps.shape[0]

    if len(axes) < n:
        raise ValueError(f"{len(axes)} axes for {n} maps.")

    norm_for = style.signed_norm if mode == "signed" else style.magnitude_norm
    cmap = {
        "signed": style.CMAP_SIGNED,
        "magnitude": style.CMAP_MAGNITUDE,
    }.get(mode) or plt.get_cmap(style.CMAP_PIXELS)

    shared = norm if norm is not None else (norm_for(maps) if shared_scale else None)
    images = []

    for k, ax in enumerate(axes):
        if k >= n:
            ax.set_visible(False)
            continue

        panel_norm = shared if shared is not None else norm_for(maps[k])
        images.append(
            ax.imshow(maps[k], cmap=cmap, norm=panel_norm, interpolation="nearest")
        )
        style.bare(ax)
        if titles and k < len(titles):
            ax.set_title(titles[k], fontsize=title_fontsize, color=style.INK_SECONDARY)

    return images


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

    Creates the grid, then hands off to :func:`draw_maps`. For a still figure
    in a notebook; inside an animation, place the axes yourself and call
    :func:`draw_maps` directly.

    Args:
        fig: Target figure; its existing axes are not cleared.
        maps: Array of shape ``(C, H, W)``.
        ncols: Panels per row.
        mode: Encoding, as in :func:`draw_matrix`.
        titles: Per-panel labels.
        suptitle: Figure-level heading.
        shared_scale: See :func:`draw_maps`.

    Returns:
        The list of images created.
    """
    maps = np.asarray(maps)
    n = maps.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    axes = fig.subplots(nrows, ncols, squeeze=False)
    images = draw_maps(
        [axes[k // ncols][k % ncols] for k in range(nrows * ncols)],
        maps, mode=mode, titles=titles, shared_scale=shared_scale,
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=style.INK_PRIMARY, x=0.02, ha="left")

    return images


# --------------------------------------------------------------------------
# Flow between panels
#
# A pipeline figure is read as "this, then this, then this", and what carries
# that reading is the connector between panels, not the panels. Both helpers
# below position themselves from the axes' *realised* figure coordinates, so
# a retuned layout moves them along with the panels instead of leaving them
# stranded — which is what happens with hardcoded figure coordinates.
# --------------------------------------------------------------------------

def _gap_centre(left_ax, right_ax, y: float | None) -> tuple[float, float]:
    """Centre of the gap between two panels, in figure coordinates."""
    left, right = left_ax.get_position(), right_ax.get_position()
    if y is None:
        y = (left.y0 + left.y1) / 2
    return (left.x1 + right.x0) / 2, y


def _gap(first_ax, second_ax) -> tuple[str, float, float, float]:
    """Orientation and extent of the gap between two panels.

    Returns ``(orientation, centre_x, centre_y, half_length)``. The larger of
    the two separations wins, so a pipeline reads left-to-right when its
    panels sit side by side and top-to-bottom when they are stacked — which
    is exactly the difference between the notebook layout and the feed cut.
    """
    a, b = first_ax.get_position(), second_ax.get_position()
    horizontal = b.x0 - a.x1
    vertical = a.y0 - b.y1

    if horizontal >= vertical:
        return "horizontal", (a.x1 + b.x0) / 2, (a.y0 + a.y1) / 2, horizontal / 2
    return "vertical", (a.x0 + a.x1) / 2, (a.y0 + b.y1) / 2, vertical / 2


def glyph_between(
    fig,
    left_ax,
    right_ax,
    symbol: str,
    y: float | None = None,
    fontsize: float = 16,
    color: str | None = None,
):
    """Set a mathematical operator in the gap between two panels.

    For the terms of an expression laid out as panels — ``patch ⊙ kernel =
    products``. Use :func:`arrow_between` instead when the relation is
    "becomes" rather than "combined with".
    """
    x, y = _gap_centre(left_ax, right_ax, y)
    return fig.text(
        x, y, symbol, ha="center", va="center",
        fontsize=fontsize, color=color or style.INK_MUTED,
    )


def arrow_between(
    fig,
    from_ax,
    to_ax,
    label: str | None = None,
    y: float | None = None,
    x: float | None = None,
    fontsize: float = 9,
    color: str | None = None,
    label_gap: float | None = None,
    shrink: float = 0.30,
):
    """Draw a labelled flow arrow from one panel to the next.

    The label names the operation that turns the first panel into the second
    — ``conv 5×5``, ``ReLU``, ``max-pool 2×2``. Naming it on the arrow rather
    than in the caption is what lets a viewer read a pipeline figure without
    the surrounding prose.

    The direction is taken from where the panels actually sit: side by side
    gives a left-to-right arrow with the label above it, stacked gives a
    downward arrow with the label beside it. The same call therefore works for
    a wide notebook figure and for the stacked feed cut of the same pipeline.

    Args:
        fig: Figure to draw on.
        from_ax: Panel the arrow leaves.
        to_ax: Panel it enters.
        label: Text set just clear of the arrow.
        y: Override the height of a horizontal arrow. Defaults to the vertical
            centre of ``from_ax`` — worth passing explicitly when the two
            panels differ in height, so the arrow sits on the band's midline
            rather than on one panel's.
        x: Override the horizontal position of a vertical arrow, likewise. Pass
            ``0.5`` to centre it on the figure when the panels it joins are one
            column of a wider row.
        fontsize: Size of the label.
        color: Ink for arrow and label; defaults to the theme's muted ink,
            because the connector is structure, not data.
        label_gap: Offset of the label from the arrow, in figure coordinates.
            Defaults to a value chosen per orientation — figure coordinates
            are fractions of two different lengths, so one number cannot serve
            both directions on a canvas that is not square.
        shrink: Fraction of the gap left clear at each end, so the arrow does
            not touch the panels it connects.
    """
    orientation, centre_x, centre_y, half = _gap(from_ax, to_ax)
    if orientation == "horizontal" and y is not None:
        centre_y = y
    if orientation == "vertical" and x is not None:
        centre_x = x
    color = color or style.INK_MUTED
    inset = half * shrink

    if orientation == "horizontal":
        gap = 0.020 if label_gap is None else label_gap
        start = (centre_x - half + inset, centre_y)
        end = (centre_x + half - inset, centre_y)
        label_xy, ha, va = (centre_x, centre_y + gap), "center", "bottom"
    else:
        # Arrows point down the page: the source panel is the one above.
        gap = 0.012 if label_gap is None else label_gap
        start = (centre_x, centre_y + half - inset)
        end = (centre_x, centre_y - half + inset)
        label_xy, ha, va = (centre_x + gap, centre_y), "left", "center"

    fig.add_artist(FancyArrowPatch(
        start, end,
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11,
        linewidth=1.1, color=color, zorder=4,
    ))

    if label:
        fig.text(*label_xy, label, ha=ha, va=va, fontsize=fontsize, color=color)
