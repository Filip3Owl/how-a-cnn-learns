"""Shared visual language for every figure and animation in this project.

Every notebook imports from here so the whole series reads as one system. The
encoding rules below are what make the animations legible:

* **Signed quantities** (weights, gradients, error signals) use a *diverging*
  blue-to-red ramp with a neutral midpoint, always symmetric about zero.
  A weight of ``-0.4`` and one of ``+0.4`` must be equally saturated, or the
  viewer reads a bias that is not in the data.
* **Unsigned magnitudes** (post-ReLU activations, feature-map energy) use a
  *sequential* single-hue blue ramp running from the surface colour outward.
* **Raw imagery** (MNIST pixels) stays greyscale. It is a photograph, not an
  encoding, and colouring it competes with the encoded panels beside it.
* **Curves over time** (loss, accuracy) take categorical slots in fixed order.

Never cycle categorical hues: slot 1 is always the same entity across every
notebook, so the reader learns the colour once.

Themes
------
Two themes are supported, ``"light"`` and ``"dark"``. **The dark theme is
selected, not flipped.** Three things genuinely reverse rather than invert:

* the sequential ramp runs dark-to-light, because "near zero" must recede
  toward the surface — on a dark page that means dark, not light;
* the diverging ramp puts its *dark* neutral in the middle and brightens
  toward both poles, so magnitude still reads as visual weight;
* MNIST digits render bright-on-dark, which is how the pixels are actually
  stored, rather than the inverted ink-on-paper of the light theme.

Calling :func:`use_project_style` rebinds the module-level colour names below,
so ``style.SURFACE`` and friends always refer to the active theme. Read them
at call time (``style.INK_PRIMARY`` inside a function), never capture them in
a default argument, which would freeze the value at import.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

__all__ = [
    "use_project_style",
    "bare",
    "signed_norm",
    "magnitude_norm",
    "THEME",
]

# --------------------------------------------------------------------------
# Theme definitions
# --------------------------------------------------------------------------

_LIGHT = {
    # Categorical hues in fixed slot order. Assign by entity, never by rank.
    "CATEGORICAL": (
        "#2a78d6",  # 1 blue
        "#008300",  # 2 green
        "#e87ba4",  # 3 magenta
        "#eda100",  # 4 yellow
        "#1baf7a",  # 5 aqua
        "#eb6834",  # 6 orange
        "#4a3aa7",  # 7 violet
        "#e34948",  # 8 red
    ),
    # Sequential: light (near zero) to dark (high magnitude).
    "SEQUENTIAL": (
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b",
    ),
    # Diverging: dark blue -> light neutral -> dark red.
    "DIVERGING": (
        "#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a58f", "#d03b3b",
        "#8f1f1f",
    ),
    "PIXELS": "gray_r",   # dark ink on a light page, as a digit is written
    "SURFACE": "#fcfcfb",
    "INK_PRIMARY": "#0b0b0b",
    "INK_SECONDARY": "#52514e",
    "INK_MUTED": "#8a8983",
    "GRID": "#e3e2de",
    "FOCUS": "#0b0b0b",
    "EMPTY": "#dcdad2",
}

_DARK = {
    # The same eight hues, re-stepped for a dark surface. Validated as a set:
    # lightness band, chroma floor, CVD separation, and >= 3:1 contrast.
    "CATEGORICAL": (
        "#3987e5",  # 1 blue
        "#008300",  # 2 green
        "#d55181",  # 3 magenta
        "#c98500",  # 4 yellow
        "#199e70",  # 5 aqua
        "#d95926",  # 6 orange
        "#9085e9",  # 7 violet
        "#e66767",  # 8 red
    ),
    # Sequential REVERSED: near-zero recedes toward the dark surface, and
    # magnitude brightens. A light-to-dark ramp on a dark page would make the
    # strongest activations vanish into the background.
    "SEQUENTIAL": (
        "#161d2b", "#12203c", "#0d366b", "#184f95", "#256abf", "#2a78d6",
        "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
        "#cde2fb",
    ),
    # Diverging re-stepped: a DARK neutral midpoint, brightening toward both
    # poles, so distance from zero still reads as visual weight.
    "DIVERGING": (
        "#9ec5f4", "#5598e7", "#2563a8", "#383835", "#a53a3a", "#e34948",
        "#f0a0a0",
    ),
    "PIXELS": "gray",     # bright digit on a dark page — how MNIST is stored
    "SURFACE": "#131312",
    "INK_PRIMARY": "#ffffff",
    "INK_SECONDARY": "#c3c2b7",
    "INK_MUTED": "#8a8983",
    "GRID": "#33322f",
    "FOCUS": "#ffffff",
    "EMPTY": "#212120",
}

THEMES = {"light": _LIGHT, "dark": _DARK}

#: Name of the currently applied theme.
THEME = "light"

# Colour names bound by :func:`use_project_style`. Declared here so importing
# the module without applying a style still yields sensible light-theme values.
CATEGORICAL = _LIGHT["CATEGORICAL"]
SEQUENTIAL = _LIGHT["SEQUENTIAL"]
CMAP_PIXELS = _LIGHT["PIXELS"]
SURFACE = _LIGHT["SURFACE"]
INK_PRIMARY = _LIGHT["INK_PRIMARY"]
INK_SECONDARY = _LIGHT["INK_SECONDARY"]
INK_MUTED = _LIGHT["INK_MUTED"]
GRID = _LIGHT["GRID"]

#: Outline for "what is being read right now" markers. Deliberately a neutral
#: ink rather than a status red: an annotation drawn on top of the diverging
#: ramp must not collide with the ramp's own red pole, or the highlight
#: vanishes exactly where the signal is strongest.
FOCUS = _LIGHT["FOCUS"]

#: Fill for cells that hold no value yet (not computed, masked out). Must read
#: as "nothing here", distinct from the diverging ramp's neutral midpoint,
#: which means a genuine zero.
EMPTY = _LIGHT["EMPTY"]

#: Reserved status colours — never reused as an extra series. Fixed across
#: themes by design; they carry meaning, not identity.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

CMAP_MAGNITUDE = LinearSegmentedColormap.from_list(
    "cnnviz_magnitude_light", _LIGHT["SEQUENTIAL"], N=256
)
CMAP_SIGNED = LinearSegmentedColormap.from_list(
    "cnnviz_signed_light", _LIGHT["DIVERGING"], N=256
)


# --------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------

def signed_norm(data: np.ndarray, robust: bool = True) -> TwoSlopeNorm:
    """Return a zero-centred norm so +x and -x are equally saturated.

    Args:
        data: The array being coloured.
        robust: Clip the scale to the 99th percentile of ``|data|``. A single
            outlier weight otherwise flattens every other cell to the neutral
            midpoint — the most common way a weight animation goes blank.

    Returns:
        A ``TwoSlopeNorm`` centred exactly on zero.
    """
    magnitude = np.abs(np.asarray(data, dtype=float))
    limit = float(np.percentile(magnitude, 99) if robust else magnitude.max())
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1e-8
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def magnitude_norm(data: np.ndarray, robust: bool = True) -> Normalize:
    """Return a 0-anchored norm for unsigned magnitudes.

    The floor is pinned at zero rather than ``data.min()`` so that "no
    activation" is always the palest colour. Letting the floor float makes a
    dead feature map look identical to an active one across frames.
    """
    values = np.asarray(data, dtype=float)
    ceiling = float(np.percentile(values, 99) if robust else values.max())
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        ceiling = 1e-8
    return Normalize(vmin=0.0, vmax=ceiling)


# --------------------------------------------------------------------------
# Applying a theme
# --------------------------------------------------------------------------

def use_project_style(theme: str = "light") -> None:
    """Apply project-wide matplotlib defaults for ``theme``.

    Call once per notebook, before creating any figure. Rebinds the module's
    colour names, so ``style.SURFACE`` and the ``CMAP_*`` colormaps refer to
    the active theme afterwards.

    Args:
        theme: ``"light"`` or ``"dark"``.

    Raises:
        ValueError: If ``theme`` is not a known theme.
    """
    if theme not in THEMES:
        raise ValueError(
            f"Unknown theme {theme!r}; expected one of {sorted(THEMES)}."
        )

    tokens = THEMES[theme]
    magnitude = LinearSegmentedColormap.from_list(
        f"cnnviz_magnitude_{theme}", tokens["SEQUENTIAL"], N=256
    )
    signed = LinearSegmentedColormap.from_list(
        f"cnnviz_signed_{theme}", tokens["DIVERGING"], N=256
    )

    for cmap in (magnitude, signed):
        try:
            mpl.colormaps.register(cmap)
        except ValueError:
            pass  # already registered on notebook re-run

    globals().update(
        THEME=theme,
        CATEGORICAL=tokens["CATEGORICAL"],
        SEQUENTIAL=tokens["SEQUENTIAL"],
        CMAP_PIXELS=tokens["PIXELS"],
        SURFACE=tokens["SURFACE"],
        INK_PRIMARY=tokens["INK_PRIMARY"],
        INK_SECONDARY=tokens["INK_SECONDARY"],
        INK_MUTED=tokens["INK_MUTED"],
        GRID=tokens["GRID"],
        FOCUS=tokens["FOCUS"],
        EMPTY=tokens["EMPTY"],
        CMAP_MAGNITUDE=magnitude,
        CMAP_SIGNED=signed,
    )

    mpl.rcParams.update({
        # Surfaces
        "figure.facecolor": tokens["SURFACE"],
        "axes.facecolor": tokens["SURFACE"],
        "savefig.facecolor": tokens["SURFACE"],

        # Recessive axes and grid — the data carries the ink, not the frame.
        "axes.edgecolor": tokens["GRID"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": tokens["GRID"],
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,

        # Text wears text tokens, never a series colour.
        "text.color": tokens["INK_PRIMARY"],
        "axes.labelcolor": tokens["INK_SECONDARY"],
        "xtick.color": tokens["INK_SECONDARY"],
        "ytick.color": tokens["INK_SECONDARY"],
        "axes.titlecolor": tokens["INK_PRIMARY"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",  # DejaVu has no "medium"; avoids a warning
        "axes.titlelocation": "left",
        "axes.titlepad": 12,

        # Marks
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "axes.prop_cycle": mpl.cycler(color=list(tokens["CATEGORICAL"])),

        # Legend sits on the surface without a competing box.
        "legend.frameon": False,
        "legend.fontsize": 10,

        # GIF frames are raster: fix DPI so frame sizes never drift mid-animation.
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": None,  # bbox="tight" resizes per frame and jitters GIFs
        "figure.constrained_layout.use": True,

        "image.interpolation": "nearest",  # never blur a pixel grid
        "image.cmap": magnitude.name,
    })


def bare(ax) -> None:
    """Strip an axis to nothing — for image panels, where ticks are noise."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
