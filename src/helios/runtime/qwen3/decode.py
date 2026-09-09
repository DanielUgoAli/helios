import logging
import time
from dataclasses import dataclass

import torch

from helios.runtime.prefix_cache import PrefixCacheHit
from helios.runtime.qwen3.cache import BatchedKVCache, KVCache
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.qwen3.paged_cache import KVPagePool, PagedBatchCache, PagedKVCache
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
    cache: KVCache | PagedKVCache


@dataclass
class PrefillResult:
    cache: KVCache | PagedKVCache
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
        self.page_pool: KVPagePool | None = None
        self._compiled_decode = (
            torch.compile(
                model.decode_forward,
                backend="inductor",
                dynamic=True,
                fullgraph=True,
                mode="default",
            )
            if torch_compile
            else None
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
        try:
            decoded = self.decode(
                prefill, eos_token_id, sampling, request_id=request_id
            )
        except Exception:
            self.release_cache(prefill.cache)
            raise
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
        cache = (
            PagedKVCache(self.page_pool, capacity)
            if self.page_pool is not None
            else KVCache(self.model.config, capacity, device=self.device)
        )
        try:
            cached_blocks = prefix_hit.blocks if prefix_hit is not None else ()
            if prefix_hit is not None:
                if any(
                    len(block.tokens) != block.snapshot.length
                    for block in cached_blocks
                ):
                    raise ValueError(
                        "Prefix-cache token and KV block lengths do not match."
                    )
                cached_tokens = tuple(
                    token for block in cached_blocks for token in block.tokens
                )
                if len(cached_tokens) > len(input_ids):
                    raise ValueError(
                        "Prefix-cache hit is longer than the request prompt."
                    )
                if tuple(input_ids[: len(cached_tokens)]) != cached_tokens:
                    raise ValueError(
                        "Prefix-cache hit does not match the request tokens."
                    )
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
                forward_logits = self._forward_cache(token_tensor, cache)
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
        except Exception:
            self.release_cache(cache)
            raise

    def decode_caches(
        self, caches: list[KVCache | PagedKVCache], token_ids: list[int]
    ) -> torch.Tensor:
        if not caches or len(caches) != len(token_ids):
            raise ValueError("Every request cache needs exactly one pending token.")
        tokens = torch.tensor(
            token_ids, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        self.model.eval()
        with torch.inference_mode():
            if isinstance(caches[0], PagedKVCache):
                cache = PagedBatchCache(caches)
                cache.prepare(1)
                slots = tuple(range(len(caches)))
                return self.model(
                    tokens,
                    cache=cache,
                    position_ids=cache.slot_lengths(slots).unsqueeze(1),
                    cache_slots=slots,
                )[:, -1, :]
            if self._compiled_decode is not None:
                return self._decode_dense(tokens, caches)[:, -1, :]
            if len(caches) == 1:
                return self.model(tokens, cache=caches[0])[:, -1, :]

            cache = BatchedKVCache(caches)
            slots = list(range(len(caches)))
            positions = cache.slot_lengths(slots).unsqueeze(1)
            return self.model(
                tokens,
                cache=cache,
                position_ids=positions,
                cache_slots=slots,
            )[:, -1, :]

    def _decode_dense(
        self, tokens: torch.Tensor, caches: list[KVCache]
    ) -> torch.Tensor:
        BatchedKVCache(caches)
        lengths = [cache.length for cache in caches]
        if any(
            length >= cache.capacity
            for length, cache in zip(lengths, caches, strict=True)
        ):
            raise ValueError("Decode would exceed request KV capacity.")
        key_length = max(lengths) + 1
        positions = torch.tensor(lengths, device=self.device).unsqueeze(1)
        layers = []
        for layer_index in range(self.model.config.n_layers):
            if len(caches) == 1:
                entry = caches[0]._layers[layer_index]
                layers.append(
                    (entry.keys[:, :, :key_length], entry.values[:, :, :key_length])
                )
                continue
            reference = caches[0]._layers[layer_index].keys
            keys = reference.new_zeros(
                len(caches), reference.shape[1], key_length, reference.shape[3]
            )
            values = torch.zeros_like(keys)
            for row, cache in enumerate(caches):
                entry = cache._layers[layer_index]
                keys[row, :, : cache.length].copy_(entry.keys[0, :, : cache.length])
                values[row, :, : cache.length].copy_(entry.values[0, :, : cache.length])
            layers.append((keys, values))
        mask = None
        if len(set(lengths)) > 1:
            mask = (torch.arange(key_length, device=self.device)[None, :] <= positions)[
                :, None, None, :
            ]
        logits = self._compiled_decode(tokens, tuple(layers), positions, mask)
        for row, cache in enumerate(caches):
            if len(caches) > 1:
                for entry, (keys, values) in zip(cache._layers, layers, strict=True):
                    entry.keys[0, :, cache.length].copy_(keys[row, :, cache.length])
                    entry.values[0, :, cache.length].copy_(values[row, :, cache.length])
            cache.advance(1)
        return logits

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
                    forward_logits = self.decode_caches([cache], [token_id]).unsqueeze(
                        1
                    )
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

    def _forward_cache(
        self, tokens: torch.Tensor, cache: KVCache | PagedKVCache
    ) -> torch.Tensor:
        if isinstance(cache, PagedKVCache):
            batch = PagedBatchCache([cache])
            batch.prepare(tokens.shape[1])
            return self.model(tokens, cache=batch)
        return self.model(tokens, cache=cache)

    @staticmethod
    def release_cache(cache: KVCache | PagedKVCache) -> None:
        if isinstance(cache, PagedKVCache):
            cache.close()

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
