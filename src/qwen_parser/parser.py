"""Batch string parser using a locally-hosted Qwen3.5-9B MLX model.

Typical usage::

    from qwen_parser import QwenParser, InferenceConfig

    cfg = InferenceConfig(temperature=0.3, top_p=0.85)
    parser = QwenParser(cfg)

    items: list[str] = ["text one", "text two", "text three"]
    results: list[str] = parser.parse_batch(items)
    for original, parsed in zip(items, results):
        print(f"--- INPUT ---\\n{original}")
        print(f"--- OUTPUT ---\\n{parsed}\\n")
"""

from __future__ import annotations

from typing import Generator

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .config import InferenceConfig


class QwenParser:
    """Loads a quantised Qwen3.5-9B MLX model once and exposes batch parsing.

    The model and tokeniser are loaded eagerly on construction so that
    repeated calls to :meth:`parse_batch` / :meth:`parse` do not pay the
    startup cost each time.

    Args:
        config: :class:`InferenceConfig` controlling model path and all
            sampling hyper-parameters.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._model, self._tokenizer = load(self.config.model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> str:
        """Parse a single string and return the model's response.

        Args:
            text: The raw string to parse.

        Returns:
            The model's generated response as a single string.
        """
        prompt = self._build_prompt(text)
        return "".join(self._stream(prompt))

    def parse_batch(self, items: list[str]) -> list[str]:
        """Parse each string in *items* and return results in the same order.

        Args:
            items: List of raw strings to process.

        Returns:
            List of model responses, one per input string.
        """
        return [self.parse(item) for item in items]

    def stream_parse(self, text: str) -> Generator[str, None, None]:
        """Stream the model's response for a single string token-by-token.

        Args:
            text: The raw string to parse.

        Yields:
            Individual text chunks as they are produced by the model.
        """
        prompt = self._build_prompt(text)
        yield from self._stream(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        """Format *text* into the Qwen3 chat template."""
        cfg = self.config
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": text})

        # apply_chat_template is available on transformers-style tokenisers
        # bundled with mlx-lm.
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _stream(self, prompt: str) -> Generator[str, None, None]:
        """Yield raw text chunks from the model for the given *prompt*."""
        cfg = self.config

        sampler = make_sampler(
            temp=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        logits_processors = make_logits_processors(
            repetition_penalty=cfg.repetition_penalty,
            repetition_context_size=cfg.repetition_context_size,
        )

        for chunk in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            yield chunk.text
