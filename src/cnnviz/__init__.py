"""cnnviz — visual explanations of how a convolutional network learns.

The shared library behind the notebook series. Notebooks import from here so
that explanation lives in prose and every figure speaks one visual language.
"""

from cnnviz import animate, data, layers, panels, results, style, text
from cnnviz.style import use_project_style
from cnnviz.text import set_language

__version__ = "0.2.0"

__all__ = [
    "animate", "data", "layers", "panels", "results", "style", "text",
    "use_project_style", "set_language",
]
