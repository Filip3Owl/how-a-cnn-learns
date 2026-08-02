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
        # Notebook 03 — the loss landscape
        "ce_title": "Cross-entropy: the price of being confidently wrong",
        "ce_x": "probability given to the right answer",
        "ce_confident_right": "confident and right",
        "ce_unsure": "a guess",
        "ce_confident_wrong": "confident and wrong",
        "ce_chance": "chance, 1 in 10",
        "features_title": "Two numbers per digit",
        "features_subtitle": "Each dot is one image, placed by its edge composition.",
        "feature_x": "share of edge energy at 45°",
        "feature_y": "share of edge energy at 90°",
        "digit_class": "digit {d}",
        "boundary": "decision boundary",
        "correct_share": "{p}% correct",
        "slice_title": "The loss along one weight, the other held still",
        "slice_cut": "the cut above",
        "w1_axis": "w₁  (45° edges)",
        "w2_axis": "w₂  (90° edges)",
        "loss_axis": "loss",
        "step_axis": "step",
        "tangent": "tangent",
        "slope_is_gradient": "slope = {v}",
        "held_at": "w₂ held at {v}",
        "landscape_title": "The whole landscape, both weights at once",
        "minimum_here": "minimum",
        "steepest_title": "The gradient is the steepest way down — measured",
        "steepest_probe": "direction, degrees",
        "steepest_drop": "loss drop per unit step",
        "steepest_measured": "steepest of {n} probes",
        "steepest_negative_gradient": "−gradient",
        "descent_title": "Gradient descent on a real loss surface",
        "descent_subtitle": (
            "Every step moves against the gradient, scaled by the learning rate."
        ),
        "descent_title_short": "How a network finds its weights",
        "descent_subtitle_short": "Downhill, one step at a time.",
        "step_of": "step {k} of {n}",
        "loss_now": "loss {v}",
        "descent_cap_start": (
            "Two weights, set badly on purpose. The surface is the actual loss."
        ),
        "descent_cap_gradient": (
            "The arrow is the negative gradient: the direction of steepest descent."
        ),
        "descent_cap_steep": "Steep ground, long steps — most of the drop happens here.",
        "descent_cap_flat": "Near the bottom the gradient shortens, and so do the steps.",
        "descent_cap_end": "Settled. No one chose these weights; the slope did.",
        # Short captions for the portrait cut. A caption is set across the full
        # canvas width, so it overruns sooner than a title does — and it is the
        # line that changes every frame, so it overruns unnoticed.
        "descent_cap_start_short": "Two weights, set badly on purpose.",
        "descent_cap_gradient_short": "The arrow is −gradient: steepest descent.",
        "descent_cap_steep_short": "Steep ground, long steps.",
        "descent_cap_flat_short": "Flatter ground, shorter steps.",
        "descent_cap_end_short": "Settled. The slope chose these weights.",
        "lr_title": "One number decides whether it ever arrives",
        "lr_subtitle": "Same surface, same start, three learning rates.",
        "lr_label": "rate {v}",
        "lr_too_small": "too small — still travelling when the budget ran out",
        "lr_right": "about right — lands and stays",
        "lr_too_large": "too large — overshoots repeatedly, arrives anyway",
        "lr_diverges": "past {v} it stops arriving at all",
        "barrier_title": "The bowl was the easy case",
        "barrier_subtitle": (
            "Straight line between two separately trained networks, both ~93% correct."
        ),
        "barrier_x": "position along the line from A to B",
        "barrier_solution": "network {name}",
        "barrier_peak": "{v}× the worse endpoint",
        "accuracy_axis": "accuracy",
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
        # Notebook 03 — a paisagem da perda
        "ce_title": "Entropia cruzada: o preço de errar com confiança",
        "ce_x": "probabilidade dada à resposta certa",
        "ce_confident_right": "confiante e certo",
        "ce_unsure": "um chute",
        "ce_confident_wrong": "confiante e errado",
        "ce_chance": "acaso, 1 em 10",
        "features_title": "Dois números por dígito",
        "features_subtitle": (
            "Cada ponto é uma imagem, posicionada pela sua composição de bordas."
        ),
        "feature_x": "fração da energia de bordas a 45°",
        "feature_y": "fração da energia de bordas a 90°",
        "digit_class": "dígito {d}",
        "boundary": "fronteira de decisão",
        "correct_share": "{p}% de acerto",
        "slice_title": "A perda ao longo de um peso, com o outro parado",
        "slice_cut": "o corte acima",
        "w1_axis": "w₁  (bordas a 45°)",
        "w2_axis": "w₂  (bordas a 90°)",
        "loss_axis": "perda",
        "step_axis": "passo",
        "tangent": "tangente",
        "slope_is_gradient": "inclinação = {v}",
        "held_at": "w₂ fixo em {v}",
        "landscape_title": "A paisagem inteira, os dois pesos de uma vez",
        "minimum_here": "mínimo",
        "steepest_title": "O gradiente é a descida mais íngreme — medido",
        "steepest_probe": "direção, em graus",
        "steepest_drop": "queda da perda por passo unitário",
        "steepest_measured": "a mais íngreme de {n} sondagens",
        "steepest_negative_gradient": "−gradiente",
        "descent_title": "Descida do gradiente numa superfície de perda real",
        "descent_subtitle": (
            "Cada passo anda contra o gradiente, "
            "escalado pela taxa de aprendizado."
        ),
        "descent_title_short": "Como uma rede acha seus pesos",
        "descent_subtitle_short": "Ladeira abaixo, um passo por vez.",
        "step_of": "passo {k} de {n}",
        "loss_now": "perda {v}",
        "descent_cap_start": (
            "Dois pesos, mal ajustados de propósito. "
            "A superfície é a perda de verdade."
        ),
        "descent_cap_gradient": (
            "A seta é o gradiente negativo: a direção de descida mais íngreme."
        ),
        "descent_cap_steep": (
            "Terreno íngreme, passos longos — quase toda a queda acontece aqui."
        ),
        "descent_cap_flat": (
            "Perto do fundo o gradiente encurta, e os passos encurtam junto."
        ),
        "descent_cap_end": "Assentou. Ninguém escolheu estes pesos; a ladeira escolheu.",
        "descent_cap_start_short": "Dois pesos, mal ajustados de propósito.",
        "descent_cap_gradient_short": "A seta é o −gradiente: a descida mais íngreme.",
        "descent_cap_steep_short": "Terreno íngreme, passos longos.",
        "descent_cap_flat_short": "Terreno mais plano, passos mais curtos.",
        "descent_cap_end_short": "Assentou. A ladeira escolheu estes pesos.",
        "lr_title": "Um único número decide se ela chega",
        "lr_subtitle": "Mesma superfície, mesmo início, três taxas de aprendizado.",
        "lr_label": "taxa {v}",
        "lr_too_small": "pequena demais — ainda a caminho quando o orçamento acabou",
        "lr_right": "no ponto — chega e fica",
        "lr_too_large": "grande demais — passa do ponto várias vezes, mas chega",
        "lr_diverges": "acima de {v} ela deixa de chegar",
        "barrier_title": "A tigela era o caso fácil",
        "barrier_subtitle": (
            "Linha reta entre duas redes treinadas em separado, "
            "ambas com ~93% de acerto."
        ),
        "barrier_x": "posição na linha de A até B",
        "barrier_solution": "rede {name}",
        "barrier_peak": "{v}× o pior extremo",
        "accuracy_axis": "acurácia",
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
