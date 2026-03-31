"""TurboQuant-style adaptive block-wise quantization for HuggingFace models."""

__version__ = "0.1.0"

from .quantizer import TurboQuantizer
from .benchmark import PerplexityBenchmark
from .gguf_convert import GGUFConverter