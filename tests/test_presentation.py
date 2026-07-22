"""Tests for the presentation layer: themes, localisation, and panels.

These guard the two things most likely to break silently. A theme that only
half-applies still renders a figure — just an unreadable one. A missing
translation still renders too, with English leaking into a Portuguese
animation. Both should fail loudly instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cnnviz import panels, style, text


@pytest.fixture(autouse=True)
def restore_defaults():
    """Themes and language are global; put them back after every test."""
    yield
    style.use_project_style("light")
    text.set_language("en")


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

def test_language_round_trip():
    text.set_language("pt-BR")
    assert text.get_language() == "pt-BR"
    assert text.t("feature_map") == "Mapa de características"


def test_unknown_language_raises():
    with pytest.raises(ValueError, match="Unknown language"):
        text.set_language("fr")


def test_missing_key_raises_rather_than_leaking_english():
    text.set_language("pt-BR")
    with pytest.raises(KeyError):
        text.t("a_key_that_does_not_exist")


def test_every_language_defines_the_same_keys():
    """A partial translation would silently render a mixed-language figure."""
    reference = set(text.STRINGS["en"])
    for language, table in text.STRINGS.items():
        assert set(table) == reference, f"{language} key set differs from en"


def test_decimal_separator_follows_language():
    text.set_language("en")
    assert text.num(1.65) == "1.65"
    text.set_language("pt-BR")
    assert text.num(1.65) == "1,65"


def test_negative_numbers_use_a_true_minus_sign():
    """U+2212, not a hyphen: it is the correct glyph and aligns with digits."""
    text.set_language("pt-BR")
    rendered = text.num(-1.6532)
    assert rendered == "−1,65"
    assert "-" not in rendered


def test_signed_formatting_and_trailing_zeros():
    text.set_language("en")
    assert text.num(2.0, places=2, signed=True) == "+2.00"
    assert text.num(0.5, places=3) == "0.500"


def test_placeholders_are_substituted():
    text.set_language("pt-BR")
    assert text.t("position_of", k=3, n=9) == "posição 3 de 9"


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

def test_unknown_theme_raises():
    with pytest.raises(ValueError, match="Unknown theme"):
        style.use_project_style("solarized")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_rebinds_tokens_and_rcparams(theme):
    style.use_project_style(theme)
    tokens = style.THEMES[theme]

    assert style.THEME == theme
    assert style.SURFACE == tokens["SURFACE"]
    assert style.INK_PRIMARY == tokens["INK_PRIMARY"]
    assert style.EMPTY == tokens["EMPTY"]
    assert matplotlib.rcParams["figure.facecolor"] == tokens["SURFACE"]


def test_dark_surface_is_actually_dark_and_light_is_light():
    style.use_project_style("dark")
    dark = _luminance(style.SURFACE)
    style.use_project_style("light")
    light = _luminance(style.SURFACE)

    assert dark < 0.1 < light


def test_sequential_ramp_reverses_direction_between_themes():
    """Near-zero must recede toward the surface, whichever surface that is.

    On light that means the ramp starts pale and darkens; on dark it must
    start dark and brighten. A ramp that ran light-to-dark on a dark page
    would hide the strongest activations in the background.
    """
    style.use_project_style("light")
    light_start, light_end = _ends(style.CMAP_MAGNITUDE)
    style.use_project_style("dark")
    dark_start, dark_end = _ends(style.CMAP_MAGNITUDE)

    assert light_start > light_end, "light ramp should darken with magnitude"
    assert dark_start < dark_end, "dark ramp should brighten with magnitude"


def test_diverging_midpoint_tracks_the_surface():
    """The neutral midpoint means "zero" and must recede, not glare."""
    for theme, expect_dark in (("light", False), ("dark", True)):
        style.use_project_style(theme)
        mid = _luminance(matplotlib.colors.to_hex(style.CMAP_SIGNED(0.5)))
        assert (mid < 0.3) is expect_dark, f"{theme} midpoint luminance {mid:.2f}"


def test_diverging_poles_are_brighter_than_the_midpoint_on_dark():
    style.use_project_style("dark")
    low = _luminance(matplotlib.colors.to_hex(style.CMAP_SIGNED(0.0)))
    mid = _luminance(matplotlib.colors.to_hex(style.CMAP_SIGNED(0.5)))
    high = _luminance(matplotlib.colors.to_hex(style.CMAP_SIGNED(1.0)))

    assert low > mid and high > mid


def test_empty_is_distinguishable_from_the_diverging_midpoint():
    """Otherwise "not computed yet" looks identical to a genuine zero."""
    for theme in ("light", "dark"):
        style.use_project_style(theme)
        empty = _luminance(style.EMPTY)
        midpoint = _luminance(matplotlib.colors.to_hex(style.CMAP_SIGNED(0.5)))
        assert abs(empty - midpoint) > 0.01, f"{theme}: EMPTY too close to zero"


def test_highlight_colour_follows_a_theme_switched_after_import():
    """Regression: the default was once captured at import and froze on light."""
    style.use_project_style("dark")
    fig, ax = plt.subplots()
    try:
        panels.highlight_window(ax, 0, 0, 2)
        stroke = ax.patches[-1].get_edgecolor()
    finally:
        plt.close(fig)

    assert matplotlib.colors.to_hex(stroke) == style.THEMES["dark"]["FOCUS"]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def test_draw_matrix_rejects_an_unknown_mode():
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="mode must be"):
            panels.draw_matrix(ax, np.zeros((3, 3)), mode="rainbow")
    finally:
        plt.close(fig)


def test_explicit_norm_overrides_the_per_panel_scale():
    """The fix that stops a weak frame rendering as vividly as a strong one."""
    fixed = style.signed_norm(np.array([-2.0, 2.0]), robust=False)
    fig, ax = plt.subplots()
    try:
        image = panels.draw_matrix(ax, np.array([[0.1, -0.1]]), mode="signed",
                                   norm=fixed)
        assert image.norm.vmax == pytest.approx(2.0)
    finally:
        plt.close(fig)


def test_masked_cells_render_as_empty_not_as_zero():
    style.use_project_style("dark")
    values = np.ma.masked_array([[1.0, 2.0]], mask=[[True, False]])
    fig, ax = plt.subplots()
    try:
        image = panels.draw_matrix(ax, values, mode="signed")
        bad = matplotlib.colors.to_hex(image.get_cmap().get_bad())
    finally:
        plt.close(fig)

    assert bad == style.THEMES["dark"]["EMPTY"]


def test_annotations_use_the_active_decimal_separator():
    text.set_language("pt-BR")
    fig, ax = plt.subplots()
    try:
        panels.draw_matrix(ax, np.array([[0.75]]), mode="signed", annotate=True)
        labels = [t.get_text() for t in ax.texts]
    finally:
        plt.close(fig)

    assert labels == ["0,75"]


def test_integral_annotations_drop_their_decimals():
    text.set_language("pt-BR")
    fig, ax = plt.subplots()
    try:
        panels.draw_matrix(ax, np.array([[-2.0, 0.0]]), mode="signed",
                           annotate=True)
        labels = sorted(t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)

    assert labels == ["0", "−2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _luminance(hex_colour: str) -> float:
    r, g, b = matplotlib.colors.to_rgb(hex_colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ends(cmap) -> tuple[float, float]:
    start = _luminance(matplotlib.colors.to_hex(cmap(0.0)))
    end = _luminance(matplotlib.colors.to_hex(cmap(1.0)))
    return start, end


# ---------------------------------------------------------------------------
# Canvas presets and video export
# ---------------------------------------------------------------------------

def test_feed_presets_are_the_documented_pixel_sizes():
    from cnnviz import formats

    assert (formats.FEED_PORTRAIT.width, formats.FEED_PORTRAIT.height) == (1080, 1350)
    assert (formats.FEED_SQUARE.width, formats.FEED_SQUARE.height) == (1080, 1080)
    assert (formats.STORY.width, formats.STORY.height) == (1080, 1920)


def test_figsize_round_trips_to_the_requested_pixels():
    """A canvas that renders at the wrong size gets cropped by the platform."""
    from cnnviz import formats

    for canvas in (formats.FEED_PORTRAIT, formats.FEED_SQUARE, formats.STORY):
        width_in, height_in = canvas.figsize
        assert round(width_in * canvas.dpi) == canvas.width
        assert round(height_in * canvas.dpi) == canvas.height


def test_phone_presets_scale_type_up():
    from cnnviz import formats

    assert formats.FEED_PORTRAIT.pt(10) == 20.0
    assert formats.WIDE.pt(10) == 10.0


def test_mp4_export_produces_a_playable_file(tmp_path):
    """Odd dimensions and the wrong pixel format both break phone playback."""
    from cnnviz import animate

    frames = [
        np.full((101, 81, 3), value, dtype=np.uint8)   # deliberately odd sizes
        for value in (10, 120, 240)
    ]
    path = animate.save_mp4(frames, tmp_path / "clip.mp4", fps=12)

    assert path.exists() and path.stat().st_size > 0

    import imageio.v2 as iio

    meta = iio.get_reader(path).get_meta_data()
    width, height = meta["size"]
    assert width % 2 == 0 and height % 2 == 0, "H.264 needs even dimensions"


def test_mp4_repeats_held_frames_to_reproduce_gif_timing(tmp_path):
    from cnnviz import animate

    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    # Frame 1 held four times as long as the base 100 ms frame.
    durations = [100.0, 400.0, 100.0]
    path = animate.save_mp4(frames, tmp_path / "held.mp4", fps=10,
                            durations=durations)

    import imageio.v2 as iio

    written = iio.get_reader(path).count_frames()
    assert written == 6, f"expected 1+4+1 frames, got {written}"


def test_gif_rejects_mismatched_frame_shapes(tmp_path):
    from cnnviz import animate

    frames = [np.zeros((10, 10, 3), np.uint8), np.zeros((12, 10, 3), np.uint8)]
    with pytest.raises(ValueError, match="share one shape"):
        animate.save_gif(frames, tmp_path / "bad.gif")


def test_hold_at_lengthens_only_the_requested_frames():
    from cnnviz import animate

    durations = animate.hold_at(5, fps=10, holds={2: 1.0}, first=0.0, last=0.0)
    assert durations == [100.0, 100.0, 1100.0, 100.0, 100.0]
