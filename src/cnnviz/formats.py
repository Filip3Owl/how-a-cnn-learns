"""Canvas presets for the platforms these animations get posted to.

A figure sized for a notebook is the wrong shape for a phone. The wide
five-panel layouts elsewhere in this project are unreadable in a feed: the
text ends up a few pixels tall and the viewer is scrolling past in under a
second. Rendering for social means a different canvas *and* a different
layout, not the same figure scaled down.

Practical notes on posting, which drive the numbers below:

* **Instagram does not accept GIF uploads.** Neither does TikTok. Both want
  MP4. X/Twitter and WhatsApp accept GIF but transcode it to video anyway;
  LinkedIn takes either. Use :func:`cnnviz.animate.save_mp4` for those
  platforms and keep the GIF for GitHub, docs and messaging. That function
  also handles the container details a feed rejects a file over — chroma
  format, a silent audio track, even dimensions.
* **Feeds autoplay muted and loop.** There is no sound to carry meaning and
  no scrubber, so every frame has to stand alone and the loop has to be
  worth watching twice.
* **Assume a small screen.** Type that is comfortable in a notebook is
  illegible at feed size; the presets carry a ``scale`` for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Canvas", "FEED_PORTRAIT", "FEED_SQUARE", "STORY", "WIDE", "FORMATS"]


@dataclass(frozen=True)
class Canvas:
    """A target canvas, in pixels, with the type scale it needs.

    Attributes:
        name: Short identifier, also used in output filenames.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        dpi: Dots per inch used to convert to matplotlib's inches.
        scale: Multiplier applied to font sizes. A phone-sized canvas needs
            roughly double the type of a notebook figure to stay legible.
        note: What the preset is for.
    """

    name: str
    width: int
    height: int
    dpi: int = 110
    scale: float = 1.0
    note: str = ""

    @property
    def figsize(self) -> tuple[float, float]:
        """Size in inches, for ``plt.figure(figsize=...)``."""
        return (self.width / self.dpi, self.height / self.dpi)

    @property
    def aspect(self) -> float:
        """Width divided by height."""
        return self.width / self.height

    def pt(self, size: float) -> float:
        """Scale a font size for this canvas."""
        return size * self.scale


#: Instagram/Facebook/LinkedIn feed, portrait. The tallest a feed post is
#: allowed to be, so it occupies the most screen as the viewer scrolls — and
#: the one shape all three show uncropped. The default.
FEED_PORTRAIT = Canvas(
    "feed", 1080, 1350, scale=2.0,
    note="Instagram/Facebook/LinkedIn feed post (4:5). Most screen real estate.",
)

#: Square. Safest across platforms and never cropped in a grid preview.
FEED_SQUARE = Canvas(
    "square", 1080, 1080, scale=2.0,
    note="Square (1:1). Safe everywhere, uncropped in grid previews.",
)

#: Full-screen vertical, for stories and short-form video.
STORY = Canvas(
    "story", 1080, 1920, scale=2.2,
    note="Stories / Reels / TikTok (9:16). Keep content clear of the "
         "top and bottom ~15%, where the interface sits.",
)

#: The notebook / README canvas, for reference.
WIDE = Canvas("wide", 1265, 473, scale=1.0, note="Notebook and README figures.")

FORMATS = {c.name: c for c in (FEED_PORTRAIT, FEED_SQUARE, STORY, WIDE)}
