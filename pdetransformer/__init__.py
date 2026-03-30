"""PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations"""

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("pdetransformer")
except Exception:
    __version__ = "0.1.11"

from . import utils
from . import visualization
from . import core
from . import data
from . import metric
from . import objectives
from . import sampler
from . import callback

__all__ = [
    "utils",
    "visualization",
    "core",
    "data",
    "metric",
    "objectives",
    "sampler",
    "callback"
]
