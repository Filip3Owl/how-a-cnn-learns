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
        # Notebook 02 — stacking layers
        "stack_title": "What each layer adds",
        "stack_subtitle": (
            "One digit through conv → ReLU → pool, twice. "
            "Every panel is the actual tensor."
        ),
        "stack_title_short": "What a CNN sees, layer by layer",
        "stack_subtitle_short": "Edges first, then combinations of edges.",
        "conv_op": "conv {k}×{k}",
        "relu_op": "ReLU",
        "pool_op": "max-pool {k}×{k}",
        "pre_activation": "Pre-activation",
        "after_relu": "After ReLU",
        "after_pool": "After pooling",
        "channel": "channel {k}",
        "edges_stage": "oriented edges",
        "combos_stage": "combinations of edges",
        # One caption per stage of the stack animation.
        "stack_cap_input": "One digit, 28×28. A single channel of grey.",
        "stack_cap_conv1": (
            "Four oriented kernels, four signed maps: "
            "where an edge is, and which way it runs."
        ),
        "stack_cap_relu1": (
            "ReLU zeroes every negative cell — half of each map goes quiet."
        ),
        "stack_cap_pool1": (
            "Max-pool halves the grid, keeping the strongest cell of each 2×2."
        ),
        "stack_cap_conv2": (
            "Layer 2 reads all four maps at once: features built out of features."
        ),
        "stack_cap_relu2": (
            "Gated again — a unit fires only where its combination is present."
        ),
        "stack_cap_pool2": (
            "Sixteen input pixels wide, one cell here. Depth bought context."
        ),
        # ReLU
        "relu_title": "ReLU keeps the positive half",
        "relu_pre": "Pre-activation (signed)",
        "relu_gate": "Gate: x > 0",
        "relu_post": "After ReLU (unsigned)",
        "relu_zeroed": "{p}% of cells set to zero",
        "threshold_title": "The bias is the unit's threshold",
        "bias_label": "bias = {v}",
        "active_cells": "{n} of {m} cells fire",
        # Two convolutions collapse into one without a nonlinearity
        "collapse_title": "Two convolutions with no ReLU are one convolution",
        "collapse_stacked": "3×3, then 3×3",
        "collapse_single": "one 5×5 kernel",
        "collapse_diff": "difference",
        "collapse_equivalent": "Equivalent 5×5 kernel",
        "largest_difference": "largest difference: {v}",
        # Max pooling
        "pool_title": "Max-pool keeps the strongest of each 2×2",
        "pool_subtitle": "The winner passes forward; the other three are dropped.",
        "pool_title_short": "What max-pooling throws away",
        "pool_subtitle_short": "Four cells in, one cell out.",
        "activation_map": "Activation map",
        "pooled_map": "Pooled map",
        "pool_window": "window",
        "pool_max": "max = {v}",
        "pool_dropped": "3 of 4 cells discarded",
        "pool_gradient": "only the winner receives gradient",
        # Translation tolerance
        "shift_title": "One pixel of shift, before and after pooling",
        "shift_original": "original",
        "shift_shifted": "shifted 1 px",
        "shift_change": "change: {p}%",
        # Receptive field
        "rf_title": "What one unit sees, by depth",
        "receptive_field": "Receptive field",
        "rf_stage": "stage",
        "rf_pixels": "input pixels seen",
        "rf_unit": "one unit here",
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
        "stack_title": "O que cada camada acrescenta",
        "stack_subtitle": (
            "Um dígito através de conv → ReLU → pooling, duas vezes. "
            "Cada painel é o tensor de verdade."
        ),
        "stack_title_short": "O que uma CNN vê, camada por camada",
        "stack_subtitle_short": "Primeiro bordas, depois combinações de bordas.",
        "conv_op": "conv {k}×{k}",
        "relu_op": "ReLU",
        # "max-pool" is what the field says in Brazil; "agrupamento máximo"
        # would be a translation nobody uses.
        "pool_op": "max-pool {k}×{k}",
        "pre_activation": "Pré-ativação",
        "after_relu": "Depois da ReLU",
        "after_pool": "Depois do pooling",
        "channel": "canal {k}",
        "edges_stage": "bordas orientadas",
        "combos_stage": "combinações de bordas",
        "stack_cap_input": "Um dígito, 28×28. Um único canal de cinza.",
        "stack_cap_conv1": (
            "Quatro kernels orientados, quatro mapas com sinal: "
            "onde há uma borda e em que direção ela corre."
        ),
        "stack_cap_relu1": (
            "A ReLU zera toda célula negativa — metade de cada mapa se cala."
        ),
        "stack_cap_pool1": (
            "O max-pool corta a grade pela metade, "
            "guardando a célula mais forte de cada 2×2."
        ),
        "stack_cap_conv2": (
            "A camada 2 lê os quatro mapas de uma vez: "
            "características feitas de características."
        ),
        "stack_cap_relu2": (
            "Filtrado de novo — a unidade dispara só onde sua combinação existe."
        ),
        "stack_cap_pool2": (
            "Dezesseis pixels de entrada, uma célula aqui. "
            "A profundidade comprou contexto."
        ),
        "relu_title": "A ReLU mantém a metade positiva",
        "relu_pre": "Pré-ativação (com sinal)",
        "relu_gate": "Máscara: x > 0",
        "relu_post": "Depois da ReLU (sem sinal)",
        "relu_zeroed": "{p}% das células zeradas",
        "threshold_title": "O viés é o limiar da unidade",
        "bias_label": "viés = {v}",
        "active_cells": "{n} de {m} células disparam",
        "collapse_title": "Duas convoluções sem ReLU são uma convolução só",
        "collapse_stacked": "3×3 e depois 3×3",
        "collapse_single": "um único kernel 5×5",
        "collapse_diff": "diferença",
        "collapse_equivalent": "Kernel 5×5 equivalente",
        "largest_difference": "maior diferença: {v}",
        "pool_title": "O max-pool guarda o mais forte de cada 2×2",
        "pool_subtitle": "O vencedor segue adiante; os outros três são descartados.",
        "pool_title_short": "O que o max-pooling joga fora",
        "pool_subtitle_short": "Quatro células entram, uma sai.",
        "activation_map": "Mapa de ativação",
        "pooled_map": "Mapa após pooling",
        "pool_window": "janela",
        "pool_max": "máx = {v}",
        "pool_dropped": "3 de 4 células descartadas",
        "pool_gradient": "só o vencedor recebe gradiente",
        "shift_title": "Um pixel de deslocamento, antes e depois do pooling",
        "shift_original": "original",
        "shift_shifted": "deslocado 1 px",
        "shift_change": "mudança: {p}%",
        "rf_title": "O que uma unidade enxerga, por profundidade",
        "receptive_field": "Campo receptivo",
        "rf_stage": "estágio",
        "rf_pixels": "pixels da entrada vistos",
        "rf_unit": "uma unidade aqui",
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
