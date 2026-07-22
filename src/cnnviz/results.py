"""Single output folder for everything the notebooks produce.

Every rendered artefact — animation or still — lands in ``results/`` at the
project root, flat and predictably named, so the GIFs can be picked up and
used elsewhere without hunting through notebook directories.

Naming is ``NN_slug.gif``, where ``NN`` is the notebook number. Sorting the
folder therefore reproduces the order of the series.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = ["RESULTS_DIR", "output_path", "list_results", "write_index"]


def output_path(name: str, notebook: int | None = None) -> Path:
    """Return a path inside ``results/``, creating the folder if needed.

    Args:
        name: File name, with extension. A ``NN_`` prefix is added when
            ``notebook`` is given and the name does not already carry one.
        notebook: Notebook number this artefact belongs to.

    Returns:
        Absolute path to write to.

    Raises:
        ValueError: If ``name`` contains a path separator. Results are a flat
            folder by design; a nested path would defeat the point.
    """
    if "/" in name or "\\" in name:
        raise ValueError(
            f"results/ is a flat folder; {name!r} must not contain a separator."
        )

    if notebook is not None and not name[:2].isdigit():
        name = f"{notebook:02d}_{name}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def list_results(pattern: str = "*") -> list[Path]:
    """Every artefact currently in ``results/``, in sorted order."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(p for p in RESULTS_DIR.glob(pattern) if p.is_file()
                  and p.name != "README.md")


def write_index() -> Path:
    """Regenerate ``results/README.md`` as a browsable gallery.

    Renders each animation inline, so the folder is self-describing on GitHub
    and in any Markdown viewer. Safe to call after every render.
    """
    rows = []
    for item in list_results():
        if item.suffix.lower() == ".gif":
            size = item.stat().st_size / 1e6
            rows.append(
                f"### {item.stem.replace('_', ' ')}\n\n"
                f"![{item.stem}]({item.name})\n\n"
                f"`{item.name}` · {size:.2f} MB\n"
            )
        elif item.suffix.lower() in {".png", ".svg"}:
            rows.append(
                f"### {item.stem.replace('_', ' ')}\n\n"
                f"![{item.stem}]({item.name})\n\n`{item.name}`\n"
            )
        elif item.suffix.lower() == ".mp4":
            size = item.stat().st_size / 1e6
            rows.append(
                f"### {item.stem.replace('_', ' ')} (video)\n\n"
                f"[{item.name}]({item.name}) · {size:.2f} MB — "
                "post this to Instagram or TikTok; neither accepts GIF.\n"
            )

    stamp = _dt.date.today().isoformat()
    body = (
        "# Results\n\n"
        "Every animation and figure produced by the notebook series, in one "
        "folder. Files are named `NN_slug`, where `NN` is the notebook that "
        "produced them.\n\n"
        f"_Regenerated {stamp} by `cnnviz.results.write_index()`._\n\n"
        "---\n\n" + "\n".join(rows)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "README.md"
    path.write_text(body, encoding="utf-8")
    return path
