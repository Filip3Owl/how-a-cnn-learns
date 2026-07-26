"""GIF authoring utilities.

The animations are the deliverable, so this module optimises for the things
that make a GIF feel authored rather than dumped out of a training loop:

* **Fixed canvas.** Every frame is rendered at an identical pixel size. The
  usual cause of a "jittering" GIF is ``bbox_inches="tight"``, which crops each
  frame to its own content and so resizes the canvas whenever a tick label
  changes width. We never use it.
* **One shared colour palette.** Quantising each frame independently makes the
  palette drift between frames, which shows up as a faint shimmer across flat
  areas. A single palette computed from the whole animation removes it — and
  because our figures are flat vector-style graphics, 96 colours is plenty.
* **Variable timing.** Uniform frame duration wastes the viewer's attention on
  transit and gives them no time at the moments that matter. Key frames can be
  held; see :func:`hold_at`.
* **Deterministic output.** Frames are written in order from an explicit list,
  never from filesystem glob order, which sorts ``frame_10`` before ``frame_2``.

Compression note: quantising to a shared palette *and* letting the encoder
write only the changed rectangle of each frame takes a typical animation here
from ~1.8 MB to ~0.5 MB. Dithering is deliberately disabled — it adds noise
that both hurts flat-graphics fidelity and roughly doubles the file size.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

__all__ = [
    "ease_in_out",
    "ease_out",
    "figure_to_frame",
    "hold_at",
    "save_gif",
    "save_mp4",
    "animate",
]

#: Default colour count for the shared palette. Flat graphics with a couple of
#: continuous ramps quantise cleanly here; raise it only if a ramp visibly bands.
DEFAULT_COLORS = 96


# --------------------------------------------------------------------------
# Easing
# --------------------------------------------------------------------------

def ease_in_out(t: float) -> float:
    """Smoothstep easing on ``t`` in [0, 1]. Slow, fast, slow."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    """Decelerating easing on ``t`` in [0, 1]. Fast start, gentle landing.

    The right choice for a value settling into place — a weight converging,
    a bar growing to its final height.
    """
    t = min(max(t, 0.0), 1.0)
    return 1.0 - math.pow(1.0 - t, 3.0)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def hold_at(
    n_frames: int,
    fps: int = 12,
    holds: dict[int, float] | None = None,
    first: float = 0.6,
    last: float = 1.6,
) -> list[float]:
    """Build a per-frame duration list in milliseconds.

    Args:
        n_frames: Number of rendered frames.
        fps: Baseline rate for ordinary frames.
        holds: Extra dwell time in seconds, keyed by frame index. Use it to
            pause on the moments the prose refers to — the frame where a
            kernel first aligns with an edge, the step where the loss jumps.
            Negative indices count from the end.
        first: Seconds held on the opening frame, so the viewer can register
            the starting state before anything moves.
        last: Seconds held on the closing frame, so the result can be read
            before the loop restarts.

    Returns:
        Duration per frame in milliseconds, suitable for a GIF encoder.
    """
    base = 1000.0 / fps
    durations = [base] * n_frames
    if n_frames == 0:
        return durations

    durations[0] += first * 1000.0
    durations[-1] += last * 1000.0

    for index, seconds in (holds or {}).items():
        durations[index] += seconds * 1000.0

    return durations


# --------------------------------------------------------------------------
# Frame capture
# --------------------------------------------------------------------------

def figure_to_frame(fig: Figure) -> np.ndarray:
    """Rasterise a figure to an RGB array without touching its canvas size."""
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return buffer[..., :3].copy()


def _shared_palette(frames: Sequence[np.ndarray], colors: int) -> Image.Image:
    """Derive one palette representative of the whole animation.

    Sampling evenly across the timeline rather than using frame 0 matters: an
    animation that starts on a nearly empty canvas would otherwise allocate
    almost its entire palette to background, and band badly once the data
    appears.
    """
    step = max(len(frames) // 24, 1)
    sample = np.concatenate([frames[i] for i in range(0, len(frames), step)], axis=0)
    return Image.fromarray(sample).quantize(colors=colors, method=Image.MEDIANCUT)


def save_gif(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 12,
    durations: Sequence[float] | None = None,
    hold_last: float = 1.6,
    hold_first: float = 0.6,
    loop: int = 0,
    colors: int = DEFAULT_COLORS,
) -> Path:
    """Write frames to an optimised, looping GIF.

    Args:
        frames: RGB arrays, all of identical shape.
        path: Destination ``.gif`` path; parent directories are created.
        fps: Playback rate, used when ``durations`` is not supplied.
        durations: Explicit per-frame duration in milliseconds, e.g. from
            :func:`hold_at`. Overrides ``fps``, ``hold_first`` and ``hold_last``.
        hold_last: Seconds to freeze on the final frame before looping.
        hold_first: Seconds to freeze on the opening frame.
        loop: ``0`` loops forever.
        colors: Palette size. See :data:`DEFAULT_COLORS`.

    Returns:
        The path written.

    Raises:
        ValueError: If ``frames`` is empty, frame shapes disagree, or
            ``durations`` has the wrong length. Shape disagreement in
            particular produces a corrupt GIF if left unchecked.
    """
    if not frames:
        raise ValueError("No frames to write.")

    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"All frames must share one shape; got {sorted(shapes)}. "
            "This is almost always savefig bbox='tight' resizing each frame."
        )

    if durations is None:
        durations = hold_at(len(frames), fps=fps, first=hold_first, last=hold_last)
    elif len(durations) != len(frames):
        raise ValueError(
            f"durations has {len(durations)} entries for {len(frames)} frames."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    palette = _shared_palette(frames, colors)
    quantised = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)
        for f in frames
    ]

    quantised[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=quantised[1:],
        duration=[int(round(d)) for d in durations],
        loop=loop,
        optimize=True,
        # disposal=1 ("leave in place") lets the encoder write only the region
        # that changed between frames. disposal=2 would clear to background
        # each time and force a full-frame rewrite, roughly 8x the file size.
        disposal=1,
    )
    return path


def _add_silent_audio(path: Path) -> None:
    """Mux a silent AAC track into an existing MP4, in place.

    A video with no audio stream at all plays fine in a browser and is still
    refused at upload by a good deal of the posting chain: Instagram's
    publishing spec lists AAC, and the common schedulers validate against it.
    The track costs a couple of kilobytes and removes a whole class of "the
    file just won't upload" that is invisible until it happens.

    ``+faststart`` moves the moov atom to the front of the file in the same
    pass, so an uploader can start reading the video before it has the whole
    thing.
    """
    import subprocess

    import imageio_ffmpeg

    muxed = path.with_name(f"{path.stem}__muxed.mp4")
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-i", str(path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
            "-shortest", "-movflags", "+faststart",
            str(muxed),
        ],
        capture_output=True, text=True, check=False,
    )

    if result.returncode != 0 or not muxed.exists():
        muxed.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not add the silent audio track to "
            f"{path.name}:\n{result.stderr.strip()[-500:]}"
        )

    muxed.replace(path)


def save_mp4(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: int = 15,
    durations: Sequence[float] | None = None,
    hold_last: float = 1.6,
    hold_first: float = 0.6,
    quality: int = 9,
    audio: bool = True,
) -> Path:
    """Write frames to an H.264 MP4 that a social feed will actually accept.

    **Instagram and TikTok do not accept GIF uploads**, and X/Twitter and
    WhatsApp transcode any GIF they are given to video anyway. For anything
    destined for a social feed this is the format to post; keep the GIF for
    GitHub, documentation and messaging apps.

    What comes out is H.264 in ``yuv420p`` with a silent AAC track and even
    pixel dimensions — the combination the feeds require. Duration is worth a
    glance too: a feed video under about three seconds is rejected outright,
    so a short animation wants its frames held rather than its loop trimmed.

    Variable frame timing has no direct equivalent in a constant-rate video,
    so ``durations`` is honoured by *repeating* frames — a frame held twice as
    long simply appears twice as many times. The result matches the GIF's
    pacing exactly at the cost of a slightly larger file.

    Args:
        frames: RGB arrays, all of identical shape.
        path: Destination ``.mp4``.
        fps: Frame rate of the output video. Keep it at 24 or above; feeds
            specify a floor in the low twenties.
        durations: Per-frame durations in ms, e.g. from :func:`hold_at`.
        hold_last: Seconds held on the final frame, when ``durations`` is None.
        hold_first: Seconds held on the opening frame, likewise.
        quality: imageio quality, 0-10. 9 is visually lossless for flat
            graphics without producing an unnecessarily large file.
        audio: Mux in a silent AAC track. On by default; see
            :func:`_add_silent_audio` for why a silent track is not a
            contradiction in terms.

    Returns:
        The path written.

    Raises:
        ValueError: If frames are empty or disagree in shape.
        RuntimeError: If ``imageio-ffmpeg`` is not installed, or the audio
            track could not be muxed in.
    """
    if not frames:
        raise ValueError("No frames to write.")
    if len({f.shape for f in frames}) > 1:
        raise ValueError("All frames must share one shape.")

    # Imported here rather than at module level: the GIF path uses PIL alone,
    # and MP4 export is optional, so a missing ffmpeg must not stop the module
    # from importing.
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "MP4 export needs ffmpeg. Install it with:\n"
            "    pip install imageio-ffmpeg"
        ) from exc

    if durations is None:
        durations = hold_at(len(frames), fps=fps, first=hold_first, last=hold_last)

    # Expand held frames into repeats so a constant-rate video reproduces the
    # GIF's timing. Every frame appears at least once, however short its hold.
    frame_ms = 1000.0 / fps
    expanded: list[np.ndarray] = []
    for frame, duration in zip(frames, durations, strict=True):
        expanded.extend([frame] * max(int(round(duration / frame_ms)), 1))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # H.264 requires even pixel dimensions; crop a stray odd row or column
    # rather than letting the encoder fail or silently rescale.
    height, width = expanded[0].shape[:2]
    if height % 2 or width % 2:
        expanded = [f[: height - height % 2, : width - width % 2] for f in expanded]

    imageio.mimwrite(
        path, expanded, fps=fps, quality=quality,
        codec="libx264",
        # yuv420p is what phones and QuickTime will actually play; libx264
        # would otherwise pick yuv444p for RGB input and the file would fail
        # to open on exactly the devices it is meant for.
        pixelformat="yuv420p",
        macro_block_size=None,  # we already guarantee even dimensions above
    )

    if audio:
        _add_silent_audio(path)

    return path


def animate(
    draw: Callable[[Figure, int], None],
    n_frames: int,
    path: str | Path,
    figsize: tuple[float, float] = (8.0, 4.5),
    fps: int = 12,
    durations: Sequence[float] | None = None,
    hold_last: float = 1.6,
    hold_first: float = 0.6,
    colors: int = DEFAULT_COLORS,
    mp4: bool = False,
    progress: bool = True,
) -> Path:
    """Render an animation by calling ``draw(fig, i)`` for each frame.

    The figure is cleared and reused between frames rather than recreated, so
    the canvas size is guaranteed constant and memory stays flat over long
    animations.

    Args:
        draw: Callback that paints frame ``i`` onto the supplied figure. It
            should not create or close figures itself.
        n_frames: Number of frames to render.
        path: Destination ``.gif``.
        figsize: Canvas size in inches, fixed for the whole animation.
        fps: Playback rate.
        durations: Per-frame durations in ms; see :func:`hold_at`.
        hold_last: Seconds held on the final frame.
        hold_first: Seconds held on the opening frame.
        colors: Palette size.
        mp4: Also write an ``.mp4`` beside the GIF, for platforms that reject
            GIF uploads. See :func:`save_mp4`.
        progress: Show a tqdm progress bar — rendering hundreds of frames is
            slow enough that silence looks like a hang.

    Returns:
        The path to the GIF.
    """
    fig = plt.figure(figsize=figsize)
    # Animations position their panels explicitly. An automatic layout engine
    # would both override those positions and re-solve them on every frame,
    # letting panels shift by a pixel or two whenever a label changes width —
    # visible as jitter once the frames are played in sequence.
    fig.set_layout_engine("none")

    frames: list[np.ndarray] = []

    indices: Iterable[int] = range(n_frames)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices, desc=Path(path).stem, unit="frame", leave=False)

    try:
        for i in indices:
            fig.clear()
            draw(fig, i)
            frames.append(figure_to_frame(fig))
    finally:
        plt.close(fig)

    gif = save_gif(
        frames, path, fps=fps, durations=durations, hold_last=hold_last,
        hold_first=hold_first, colors=colors,
    )

    if mp4:
        save_mp4(
            frames, gif.with_suffix(".mp4"), fps=max(fps, 24),
            durations=durations, hold_last=hold_last, hold_first=hold_first,
        )

    return gif
