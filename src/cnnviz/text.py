"""Localised strings for rendered figures and animations.

The notebooks, code and documentation are written in English. The *rendered
output* — every label, title and caption baked into a GIF or figure — is
localised through this module, so a Brazilian-Portuguese animation can be
produced without translating the surrounding explanation.

Usage::

    from cnnviz import text
    text.set_language("pt-BR")
    text.t("feature_map")          # -> "Mapa de características"
    text.num(-1.6532, 2)           # -> "−1,65"

Number formatting is part of localisation, not decoration: pt-BR writes
``1,65`` where English writes ``1.65``. An animation that translates its
labels but leaves a full stop in every cell reads as half-finished, so
:func:`num` is used for every number drawn into a figure.

The minus sign emitted is U+2212 MINUS SIGN, not a hyphen. It is the correct
character, and it aligns with digits where a hyphen does not.
"""

from __future__ import annotations

__all__ = ["set_language", "get_language", "t", "num", "LANGUAGES"]

MINUS = "−"

#: Translations keyed by language tag, then by string id.
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # Notebook 01 — the convolution sweep
        "conv_title": "How a convolution builds a feature map",
        "conv_subtitle": (
            "A vertical-edge kernel sweeps the image; "
            "each stop produces one number."
        ),
        # Short forms for narrow canvases (phone feed). Not the long strings
        # at a smaller size — on a 1080px-wide canvas the full title wraps or
        # overruns, and a title that has to be squinted at is worse than a
        # shorter one that reads at a glance.
        "conv_title_short": "How a CNN sees an edge",
        "conv_subtitle_short": "One kernel, one number at a time.",
        "input": "Input",
        "patch": "patch",
        "kernel": "kernel",
        "products": "products",
        "feature_map": "Feature map",
        "sum": "sum",
        "position_of": "position {k} of {n}",
        "strong_edge": "strong vertical edge here",
        "weak_edge": "weak response — little vertical structure",
        # Shared
        "full_digit": "Full digit",
        "crop": "Crop",
        "static_title": "A vertical-edge detector applied to a 7",
        "geometries_title": "The same kernel under three geometries",
        "stride": "stride",
        "padding": "pad",
    },
    "pt-BR": {
        "conv_title": "Como uma convolução constrói um mapa de características",
        "conv_subtitle": (
            "Um kernel de bordas verticais percorre a imagem; "
            "cada parada produz um número."
        ),
        "conv_title_short": "Como uma CNN enxerga uma borda",
        "conv_subtitle_short": "Um kernel, um número por vez.",
        "input": "Entrada",
        "patch": "trecho",
        # "kernel" is standard in Brazilian ML usage; "núcleo" would be a
        # translation nobody in the field actually says.
        "kernel": "kernel",
        "products": "produtos",
        "feature_map": "Mapa de características",
        "sum": "soma",
        "position_of": "posição {k} de {n}",
        "strong_edge": "borda vertical forte aqui",
        "weak_edge": "resposta fraca — pouca estrutura vertical",
        "full_digit": "Dígito completo",
        "crop": "Recorte",
        "static_title": "Um detector de bordas verticais aplicado a um 7",
        "geometries_title": "O mesmo kernel sob três geometrias",
        "stride": "passo",
        "padding": "preench.",
    },
}

LANGUAGES = tuple(STRINGS)

#: Decimal separator per language.
_DECIMAL = {"en": ".", "pt-BR": ","}

_language = "en"


def set_language(language: str) -> None:
    """Select the language used by :func:`t` and :func:`num`.

    Args:
        language: One of :data:`LANGUAGES`.

    Raises:
        ValueError: If the language is not available. Failing loudly beats
            silently rendering an animation half in the wrong language.
    """
    global _language
    if language not in STRINGS:
        raise ValueError(
            f"Unknown language {language!r}; expected one of {list(LANGUAGES)}."
        )
    _language = language


def get_language() -> str:
    """The currently selected language tag."""
    return _language


def t(key: str, **kwargs) -> str:
    """Look up a localised string, substituting any ``{placeholders}``.

    Args:
        key: String id, as used in :data:`STRINGS`.
        **kwargs: Values for placeholders in the template.

    Returns:
        The localised string.

    Raises:
        KeyError: If the id is missing from the active language. Again, loud
            beats silent: a missing translation should stop the render, not
            leak English into a Portuguese animation.
    """
    table = STRINGS[_language]
    if key not in table:
        raise KeyError(f"No {_language!r} string for {key!r}.")
    return table[key].format(**kwargs) if kwargs else table[key]


def num(value: float, places: int = 2, signed: bool = False) -> str:
    """Format a number for display, using the active language's separator.

    Args:
        value: The number to render.
        places: Digits after the separator. Trailing zeros are kept, so a
            column of numbers stays visually aligned across frames.
        signed: Always show a leading sign, for quantities where the sign is
            the message.

    Returns:
        The formatted string, using U+2212 for negatives.
    """
    text = f"{abs(value):.{places}f}"
    text = text.replace(".", _DECIMAL[_language])

    if value < 0:
        return MINUS + text
    return f"+{text}" if signed else text
