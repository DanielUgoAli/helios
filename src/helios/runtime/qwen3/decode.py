import logging
import time
from dataclasses import dataclass

import torch

from helios.runtime.prefix_cache import PrefixCacheHit
from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")
PROGRESS_INTERVAL_TOKENS = 32


@dataclass
class DecodeResult:
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]
    restore_seconds: float
    restored_tokens: int
    cache: KVCache


@dataclass
class PrefillResult:
    cache: KVCache
    logits: torch.Tensor
    prefill_seconds: float
    restore_seconds: float
    restored_tokens: int


@dataclass
class DecodedTokens:
    output_ids: list[int]
    finish_reason: str
    inter_token_seconds: list[float]


class Decoder:
    def __init__(self, model: Qwen3Model, *, torch_compile: bool = False) -> None:
        self.model = model
        # Stable-slot forwards use tensor-indexed cache rows and intentionally
        # stay eager. The original uniform-cache path remains compilable.
        self._slot_forward = model
        self._forward = (
            torch.compile(
                model,
                dynamic=True,
                fullgraph=True,
                mode="default",
            )
            if torch_compile
            else model
        )

    def generate(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
        request_id: str = "internal",
    ) -> DecodeResult:
        prefill = self.prefill(
            input_ids,
            sampling,
            max_total_tokens=max_total_tokens,
            prefix_hit=prefix_hit,
        )
        decoded = self.decode(prefill, eos_token_id, sampling, request_id=request_id)
        return DecodeResult(
            output_ids=decoded.output_ids,
            finish_reason=decoded.finish_reason,
            prefill_seconds=prefill.prefill_seconds,
            inter_token_seconds=decoded.inter_token_seconds,
            restore_seconds=prefill.restore_seconds,
            restored_tokens=prefill.restored_tokens,
            cache=prefill.cache,
        )

    def prefill(
        self,
        input_ids: list[int],
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: PrefixCacheHit | None = None,
    ) -> PrefillResult:
        capacity = len(input_ids) + sampling.max_new_tokens
        if capacity > max_total_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled "
                f"limit is {max_total_tokens:,}."
            )
        # Cache capacity includes the prompt and the maximum possible continuation.
        cache = KVCache(self.model.config, capacity, device=self.device)
        cached_blocks = prefix_hit.blocks if prefix_hit is not None else ()
        if prefix_hit is not None:
            if any(
                len(block.tokens) != block.snapshot.length for block in cached_blocks
            ):
                raise ValueError(
                    "Prefix-cache token and KV block lengths do not match."
                )
            cached_tokens = tuple(
                token for block in cached_blocks for token in block.tokens
            )
            if len(cached_tokens) > len(input_ids):
                raise ValueError("Prefix-cache hit is longer than the request prompt.")
            if tuple(input_ids[: len(cached_tokens)]) != cached_tokens:
                raise ValueError("Prefix-cache hit does not match the request tokens.")
            if len(cached_tokens) == len(input_ids):
                cached_blocks = cached_blocks[:-1]

        restore_started = time.perf_counter()
        cache.restore_blocks(tuple(block.snapshot for block in cached_blocks))
        restore_seconds = time.perf_counter() - restore_started
        restored_tokens = cache.length
        token_tensor = torch.tensor(
            input_ids[cache.length :], device=self.device
        ).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            forward_logits = self._forward(token_tensor, cache=cache)
            self._validate_forward_shapes(token_tensor, forward_logits)
            self._synchronize()
            prefill_seconds = time.perf_counter() - started
        return PrefillResult(
            cache=cache,
            logits=forward_logits[:, -1, :],
            prefill_seconds=prefill_seconds,
            restore_seconds=restore_seconds,
            restored_tokens=restored_tokens,
        )

    def slot_cache(self, *, capacity: int, slots: int) -> KVCache:
        """Allocate the fixed rows used by one continuous-batching cohort."""
        return KVCache(
            self.model.config, capacity, device=self.device, batch_size=slots
        )

    def prefill_slot(
        self,
        cache: KVCache,
        slot: int,
        input_ids: list[int],
        *,
        prefix_hit: PrefixCacheHit | None = None,
    ) -> PrefillResult:
        """Prefill one new request directly into a free cache row."""
        if not input_ids:
            raise ValueError("A prompt must contain at least one token.")
        cache.clear_slot(slot)
        cached_blocks = prefix_hit.blocks if prefix_hit is not None else ()
        if prefix_hit is not None:
            if any(
                len(block.tokens) != block.snapshot.length for block in cached_blocks
            ):
                raise ValueError(
                    "Prefix-cache token and KV block lengths do not match."
                )
            cached_tokens = tuple(
                token for block in cached_blocks for token in block.tokens
            )
            if (
                len(cached_tokens) > len(input_ids)
                or tuple(input_ids[: len(cached_tokens)]) != cached_tokens
            ):
                raise ValueError("Prefix-cache hit does not match the request tokens.")
            if len(cached_tokens) == len(input_ids):
                cached_blocks = cached_blocks[:-1]

        restore_started = time.perf_counter()
        cache.restore_blocks_into_slot(
            slot, tuple(block.snapshot for block in cached_blocks)
        )
        restore_seconds = time.perf_counter() - restore_started
        restored_tokens = cache.slot_length(slot)
        token_tensor = torch.tensor(
            input_ids[restored_tokens:], device=self.device
        ).unsqueeze(0)
        if token_tensor.shape[1] < 1:
            raise RuntimeError("Prefix-cache restore left no token to prefill.")

        self.model.eval()
        with torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            logits = self._slot_forward(token_tensor, cache=cache, cache_slots=[slot])[
                :, -1, :
            ]
            self._synchronize()
            prefill_seconds = time.perf_counter() - started
        return PrefillResult(
            cache=cache,
            logits=logits,
            prefill_seconds=prefill_seconds,
            restore_seconds=restore_seconds,
            restored_tokens=restored_tokens,
        )

    def decode_slots(
        self,
        cache: KVCache,
        slots: list[int],
        token_ids: list[int],
    ) -> torch.Tensor:
        """Append one pending token for each occupied slot and return its logits."""
        if not slots or len(slots) != len(token_ids):
            raise ValueError("Every occupied slot needs exactly one pending token.")
        tokens = torch.tensor(
            token_ids, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        positions = cache.slot_lengths(slots).unsqueeze(1)
        self.model.eval()
        with torch.inference_mode():
            logits = self._slot_forward(
                tokens,
                cache=cache,
                position_ids=positions,
                cache_slots=slots,
            )
        return logits[:, -1, :]

    def decode(
        self,
        prefill: PrefillResult,
        eos_token_id: int,
        sampling: Sampling,
        *,
        request_id: str = "internal",
    ) -> DecodedTokens:
        generated: list[int] = []
        inter_token_seconds: list[float] = []
        finish_reason = "length"
        logits = prefill.logits
        cache = prefill.cache
        self.model.eval()
        with torch.inference_mode():
            for index in range(sampling.max_new_tokens):
                next_token = self._sample(logits, sampling)
                token_id = next_token.item()
                if token_id == eos_token_id:
                    finish_reason = "eos"
                    break
                generated.append(token_id)
                generated_tokens = len(generated)
                if (
                    generated_tokens == 1
                    or generated_tokens % PROGRESS_INTERVAL_TOKENS == 0
                ):
                    logger.info(
                        "generation_progress request_id=%s output_tokens=%d "
                        "max_new_tokens=%d last_token_ms=%.1f",
                        request_id,
                        generated_tokens,
                        sampling.max_new_tokens,
                        (
                            prefill.prefill_seconds
                            if index == 0
                            else inter_token_seconds[-1]
                        )
                        * 1_000,
                    )
                if index + 1 < sampling.max_new_tokens:
                    self._synchronize()
                    started = time.perf_counter()
                    forward_logits = self._forward(next_token, cache=cache)
                    self._validate_forward_shapes(next_token, forward_logits)
                    self._synchronize()
                    inter_token_seconds.append(time.perf_counter() - started)
                    logits = forward_logits[:, -1, :]
        return DecodedTokens(
            output_ids=generated,
            finish_reason=finish_reason,
            inter_token_seconds=inter_token_seconds,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _validate_forward_shapes(
        self, input_ids: torch.Tensor, logits: torch.Tensor
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise RuntimeError(
                "Decoder input must have shape [1, tokens] with at least one token."
            )
        if input_ids.dtype != torch.long:
            raise RuntimeError("Decoder input token IDs must use torch.long.")
        expected_logits = (1, 1, self.model.config.vocab_size)
        if logits.shape != expected_logits:
            raise RuntimeError(
                "Decoder logits must have shape "
                f"{expected_logits}; received {tuple(logits.shape)}."
            )

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        if sampling.temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probabilities = torch.softmax(logits / sampling.temperature, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        remove = torch.cumsum(sorted_probabilities, dim=-1) > sampling.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(-1, sampled)
