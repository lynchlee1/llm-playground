"""qwen_parser — local Qwen3.5-9B MLX batch parser."""

from .config import InferenceConfig
from .parser import QwenParser

__all__ = ["InferenceConfig", "QwenParser"]
