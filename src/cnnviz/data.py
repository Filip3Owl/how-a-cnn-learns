"""MNIST loading, cached locally and shared by every notebook.

The IDX files are fetched and parsed here directly rather than through
``torchvision.datasets``. Two reasons:

1. torchvision's mirror list still leads with ``yann.lecun.com``, which has
   served 404 for the MNIST files since the site was reorganised. It burns a
   failed request on every cold start.
2. The IDX container is about fifteen lines of ``struct`` and ``numpy``.
   Parsing it in the open suits a project whose whole premise is that nothing
   should be hidden behind a framework call.
"""

from __future__ import annotations

import gzip
import hashlib
import ssl
import struct
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

#: Filenames and their expected SHA-256, so a truncated or substituted
#: download fails loudly instead of quietly training on garbage.
FILES = {
    "train_images": (
        "train-images-idx3-ubyte.gz",
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    ),
}

# Channel statistics of the MNIST training split, used to standardise inputs.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context backed by certifi's CA bundle.

    Python installs on macOS frequently ship without a usable system CA store
    (the ``Install Certificates.command`` step is easy to miss), which makes
    every HTTPS download fail with CERTIFICATE_VERIFY_FAILED. Using certifi's
    bundle explicitly sidesteps that without weakening verification — never
    "fix" this by disabling certificate checks.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(filename: str, expected_sha256: str, target: Path) -> Path:
    """Fetch one gzipped IDX file if it is not already cached and valid."""
    path = target / filename
    if path.exists() and _sha256(path) == expected_sha256:
        return path

    target.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{filename}"
    print(f"Downloading {filename} …")

    request = urllib.request.Request(url, headers={"User-Agent": "cnnviz"})
    with urllib.request.urlopen(request, context=_ssl_context(), timeout=120) as r:
        payload = r.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {filename}.\n"
            f"  expected {expected_sha256}\n  got      {digest}\n"
            "The mirror may have changed; do not train on this file."
        )

    path.write_bytes(payload)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_idx(path: Path) -> np.ndarray:
    """Parse a gzipped IDX file into a NumPy array.

    The IDX header is a 4-byte magic number — two zero bytes, a dtype code,
    and the number of dimensions — followed by one big-endian ``int32`` per
    dimension, then the raw data. MNIST only ever uses dtype code ``0x08``
    (unsigned byte).
    """
    with gzip.open(path, "rb") as handle:
        magic = handle.read(4)
        zero, dtype_code, n_dims = magic[0] << 8 | magic[1], magic[2], magic[3]
        if zero != 0 or dtype_code != 0x08:
            raise ValueError(f"{path.name} is not an unsigned-byte IDX file.")

        shape = struct.unpack(f">{n_dims}I", handle.read(4 * n_dims))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)


def load_mnist(train: bool = True, normalize: bool = True):
    """Load MNIST as ``(images, labels)`` NumPy arrays.

    Downloads to ``data/raw/`` on first call and reuses the cache afterwards.

    Args:
        train: Load the 60k training split, else the 10k test split.
        normalize: Standardise to zero mean / unit variance. Turn this **off**
            for anything being displayed — a standardised digit has negative
            pixels and renders with a grey background. Train on normalised
            data, visualise the raw [0, 1] version.

    Returns:
        ``images`` of shape ``(N, 1, 28, 28)`` as float32, and ``labels`` of
        shape ``(N,)`` as int64.
    """
    split = "train" if train else "test"
    image_file, image_hash = FILES[f"{split}_images"]
    label_file, label_hash = FILES[f"{split}_labels"]

    images = _read_idx(_download(image_file, image_hash, DATA_DIR))
    labels = _read_idx(_download(label_file, label_hash, DATA_DIR))

    images = images.astype(np.float32) / 255.0
    images = images[:, None, :, :]  # (N, 1, 28, 28)
    labels = labels.astype(np.int64)

    if normalize:
        images = (images - MNIST_MEAN) / MNIST_STD

    return images, labels


def sample_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 1,
    seed: int = 0,
):
    """Draw a class-balanced sample, ordered 0-9.

    Useful for the "one digit of each" strips that recur throughout the
    notebooks. Ordering by class keeps the strip stable across notebooks so
    the reader can compare panels position by position.
    """
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for digit in range(10):
        candidates = np.flatnonzero(labels == digit)
        picked.extend(rng.choice(candidates, size=n_per_class, replace=False))
    index = np.array(picked)
    return images[index], labels[index]
