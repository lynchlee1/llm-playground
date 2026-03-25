"""Inference configuration for Qwen3.5-9B-MLX-Q5."""

from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    """All tunable parameters for MLX-LM generation.

    Attributes:
        model_path: Local path to the quantised MLX model directory.
        temperature: Sampling temperature. Higher = more random, 0 = greedy.
        top_p: Nucleus-sampling probability mass cutoff (0 < top_p <= 1).
        top_k: Keep only the top-k tokens during sampling (0 = disabled).
        max_tokens: Maximum number of tokens to generate per call.
        repetition_penalty: Penalise recently-seen tokens (1.0 = off).
        repetition_context_size: Window size for repetition penalty.
        system_prompt: Optional system message prepended to every request.
    """

    model_path: str = "./Qwen3.5-9B-MLX-Q5"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    system_prompt: str = (
        "You are a helpful assistant. "
        "Parse the provided text and return structured information."
    )
