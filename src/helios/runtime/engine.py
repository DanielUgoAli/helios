import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from threading import Lock

import torch

from helios.config import HeliosConfig
from helios.runtime.check import MemoryChecker
from helios.runtime.generate import GenerationResult, Generator, PrefixTrace
from helios.runtime.load import Loader
from helios.runtime.prefix_cache import PromptBlockView, describe_prompt_blocks
from helios.runtime.qwen3.cache import KVCache
from helios.runtime.qwen3.paged_cache import PagedKVCache
from helios.runtime.scheduler import Job, Scheduler
from helios.runtime.types import Sampling

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class _Request:
    input_ids: list[int]
    eos_token_id: int
    sampling: Sampling
    request_id: str


@dataclass
class _ActiveRequest:
    job: Job[_Request, GenerationResult]
    request: _Request
    cache: KVCache | PagedKVCache
    reservation_bytes: int
    output_ids: list[int]
    pending_token_id: int
    started_at: float
    prefill_seconds: float
    inter_token_seconds: list[float]
    prefix_lookup_seconds: float
    restore_seconds: float
    hit_tokens: int
    restored_tokens: int
    prompt_blocks: tuple[PromptBlockView, ...]


class Engine:
    def __init__(self, config: HeliosConfig, loader: Loader | None = None) -> None:
        loaded = (loader or Loader()).load(config)
        self.model_id = config.model_id
        self.model_revision = loaded.model_revision
        self._memory_checker = MemoryChecker(config)
        self.report = loaded.report
        self.generator = Generator(
            loaded.model,
            loaded.cache,
            prefix_cache_ttl_seconds=config.prefix_cache_ttl_seconds,
            paged_attention=config.paged_attention,
        )
        self._generation_lock = Lock()
        self._max_batch_size = config.max_batch_size
        self._active_requests: list[_ActiveRequest] = []
        self._scheduler: Scheduler[_Request, GenerationResult] = Scheduler(
            self._continuous_tick,
            max_batch_size=config.max_batch_size,
            max_queue_size=config.max_queue_size,
            batch_wait_seconds=config.batch_wait_ms / 1_000,
        )

    def update_cache_capacity(
        self,
        *,
        warmup_peak_bytes: int,
        warmup_kv_bytes: int,
    ) -> None:
        with self._generation_lock:
            if self._active_requests:
                raise RuntimeError("Cannot reprofile KV memory while requests are active.")
            if self.generator.paged_attention:
                self.generator.release_page_pool()
                torch.cuda.empty_cache()
            cache = self._memory_checker.cache(
                self.generator.decoder.model.config,
                warmup_peak_bytes=warmup_peak_bytes,
                warmup_kv_bytes=warmup_kv_bytes,
            )
            self.generator.update_cache_capacity(cache)
            self.report = replace(self.report, cache=cache)

    def prefix_cache_snapshot(self) -> dict[str, object]:
        with self._generation_lock:
            cache = self.generator.prefix_cache
            blocks = cache.blocks()
            return {
                "block_size": cache.block_size,
                "occupied_blocks": len(blocks),
                "cached_tokens": cache.token_count,
                "memory_bytes": cache.memory_bytes,
                "max_blocks": None,
                "max_memory_bytes": cache.max_memory_bytes,
                "blocks": [block.as_dict() for block in blocks],
            }

    def scheduler_snapshot(self) -> dict[str, object]:
        return self._scheduler.snapshot()

    def close(self) -> None:
        self._scheduler.close()
        with self._generation_lock:
            for active in self._active_requests:
                self.generator.decoder.release_cache(active.cache)
            self._active_requests = []
            self.generator.reserve_active_cache(0)
            if self.generator.paged_attention:
                self.generator.release_page_pool()

    def run(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str | None = None,
    ) -> GenerationResult:
        return self.enqueue(
            input_ids,
            eos_token_id,
            sampling,
            request_id=request_id,
        ).result()

    def warm_decode(self, input_ids: list[int]) -> tuple[int, tuple[int, ...]]:
        from helios.runtime.warmup import warm_decode

        with self._generation_lock:
            if self._active_requests:
                raise RuntimeError("Cannot warm decode while requests are active.")
            capacity = self.generator.cache
            return warm_decode(
                self.generator.decoder,
                input_ids,
                max_batch_size=self._max_batch_size,
                max_tokens=capacity.max_tokens,
                budget_tokens=capacity.kv_budget_bytes // capacity.bytes_per_token,
            )

    def run_warmup(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str,
    ) -> GenerationResult:
        self._validate_request(input_ids, eos_token_id, sampling)
        with self._generation_lock:
            logger.info(
                "request_running request_id=%s prompt_tokens=%d max_new_tokens=%d queue_ms=0.0",
                request_id,
                len(input_ids),
                sampling.max_new_tokens,
            )
            result = self.generator.run(
                input_ids,
                eos_token_id,
                sampling,
                request_id=request_id,
            )
            return self._finish_scheduled_request(result, 0.0, request_id)

    def enqueue(
        self,
        input_ids: list[int],
        eos_token_id: int,
        sampling: Sampling,
        request_id: str | None = None,
    ) -> Future[GenerationResult]:
        request_id = request_id or "internal"
        self._validate_request(input_ids, eos_token_id, sampling)
        logger.info("request_waiting request_id=%s", request_id)
        payload = _Request(input_ids, eos_token_id, sampling, request_id)
        return self._scheduler.enqueue(Job(payload=payload, request_ids=(request_id,)))

    def _continuous_tick(
        self, scheduler: Scheduler[_Request, GenerationResult]
    ) -> bool:
        with self._generation_lock:
            self._drop_cancelled_active()
            scheduler.peek()
            if self._active_requests:
                try:
                    self._decode_active_requests()
                except Exception as error:
                    self._fail_active_requests(error)

            self._admit_requests(scheduler)
            scheduler.set_active(tuple(active.job for active in self._active_requests))
            return bool(self._active_requests or scheduler.peek() is not None)

    def _admit_requests(self, scheduler: Scheduler[_Request, GenerationResult]) -> None:
        if len(self._active_requests) >= self._max_batch_size:
            head = scheduler.peek()
            if head is not None:
                logger.info(
                    "continuous_admission_blocked request_id=%s reason=slots "
                    "active_request_ids=%s %s",
                    head.payload.request_id,
                    [active.request.request_id for active in self._active_requests],
                    self._memory_log_fields(self._reserved_memory_bytes()),
                )
            return
        while len(self._active_requests) < self._max_batch_size:
            head = scheduler.peek()
            if head is None:
                return
            request = head.payload
            capacity = len(request.input_ids) + request.sampling.max_new_tokens
            reservation_bytes = self.generator.request_cache_bytes(capacity)
            if reservation_bytes > self.generator.kv_budget_bytes:
                job = scheduler.take(head)
                if job is not None and not job.future.done():
                    job.future.set_exception(
                        RuntimeError(
                            "The FIFO request cannot fit in the KV-cache budget."
                        )
                    )
                continue
            reserved = self._reserved_memory_bytes(extra_capacity=capacity)
            if reserved > self.generator.kv_budget_bytes:
                logger.info(
                    "continuous_admission_blocked request_id=%s reason=memory "
                    "budget_bytes=%d active_request_ids=%s %s",
                    request.request_id,
                    self.generator.cache.kv_budget_bytes,
                    [active.request.request_id for active in self._active_requests],
                    self._memory_log_fields(reserved),
                )
                return

            job = scheduler.take(head)
            if job is None:
                return
            self._start_request(job, reservation_bytes, reserved)

    def _start_request(
        self,
        job: Job[_Request, GenerationResult],
        reservation_bytes: int,
        reserved_memory_bytes: int,
    ) -> None:
        request = job.payload
        queue_seconds = time.perf_counter() - job.enqueued_at
        logger.info(
            "request_running request_id=%s prompt_tokens=%d max_new_tokens=%d queue_ms=%.1f",
            request.request_id,
            len(request.input_ids),
            request.sampling.max_new_tokens,
            queue_seconds * 1_000,
        )
        request_started = time.perf_counter()
        prefill = None
        active = None
        try:
            self.generator.reserve_active_cache(reserved_memory_bytes)
            lookup_started = time.perf_counter()
            prefix_hit = self.generator.prefix_cache.longest_prefix(request.input_ids)
            prefix_lookup_seconds = time.perf_counter() - lookup_started
            prefill = self.generator.decoder.prefill(
                request.input_ids,
                request.sampling,
                max_total_tokens=self.generator.cache.max_tokens,
                prefix_hit=prefix_hit,
            )
            active = _ActiveRequest(
                job=job,
                request=request,
                cache=prefill.cache,
                reservation_bytes=reservation_bytes,
                output_ids=[],
                pending_token_id=0,
                started_at=request_started,
                prefill_seconds=prefill.prefill_seconds,
                inter_token_seconds=[],
                prefix_lookup_seconds=prefix_lookup_seconds,
                restore_seconds=prefill.restore_seconds,
                hit_tokens=0 if prefix_hit is None else prefix_hit.length,
                restored_tokens=prefill.restored_tokens,
                prompt_blocks=describe_prompt_blocks(
                    request.input_ids,
                    self.generator.prefix_cache.block_size,
                    prefix_hit,
                ),
            )
            token_id = int(
                self.generator.decoder._sample(prefill.logits, request.sampling).item()
            )
            if self._accept_token(active, token_id, queue_seconds):
                self._active_requests.append(active)
                logger.info(
                    "continuous_admitted request_id=%s active_request_ids=%s %s",
                    request.request_id,
                    [item.request.request_id for item in self._active_requests],
                    self._memory_log_fields(self._reserved_memory_bytes()),
                )
            else:
                active = None
                prefill = None
                self.generator.reserve_active_cache(self._reserved_memory_bytes())
        except Exception as error:
            if prefill is not None:
                self.generator.decoder.release_cache(prefill.cache)
            if not job.future.done():
                job.future.set_exception(error)
            active = None
            prefill = None
            self.generator.reserve_active_cache(self._reserved_memory_bytes())

    def _decode_active_requests(self) -> None:
        tokens = [active.pending_token_id for active in self._active_requests]
        self.generator.decoder._synchronize()
        started = time.perf_counter()
        logger.info(
            "continuous_decode active_request_ids=%s %s",
            [active.request.request_id for active in self._active_requests],
            self._memory_log_fields(self._reserved_memory_bytes()),
        )
        logits = self.generator.decoder.decode_caches(
            [active.cache for active in self._active_requests], tokens
        )
        self.generator.decoder._synchronize()
        elapsed = time.perf_counter() - started

        surviving: list[_ActiveRequest] = []
        for row, active in enumerate(self._active_requests):
            token_id = int(
                self.generator.decoder._sample(
                    logits[row : row + 1], active.request.sampling
                ).item()
            )
            active.inter_token_seconds.append(elapsed)
            queue_seconds = active.started_at - active.job.enqueued_at
            if self._accept_token(active, token_id, queue_seconds):
                surviving.append(active)
        self._active_requests = surviving
        self.generator.reserve_active_cache(self._reserved_memory_bytes())

    def _accept_token(
        self, active: _ActiveRequest, token_id: int, queue_seconds: float
    ) -> bool:
        if token_id == active.request.eos_token_id:
            self._complete_request(active, "eos", queue_seconds)
            return False
        active.output_ids.append(token_id)
        if len(active.output_ids) == active.request.sampling.max_new_tokens:
            self._complete_request(active, "length", queue_seconds)
            return False
        active.pending_token_id = token_id
        return True

    def _drop_cancelled_active(self) -> None:
        surviving: list[_ActiveRequest] = []
        for active in self._active_requests:
            if active.job.future.cancelled():
                self.generator.decoder.release_cache(active.cache)
                logger.info(
                    "continuous_cancelled request_id=%s", active.request.request_id
                )
            else:
                surviving.append(active)
        self._active_requests = surviving
        self.generator.reserve_active_cache(self._reserved_memory_bytes())

    def _complete_request(
        self, active: _ActiveRequest, finish_reason: str, queue_seconds: float
    ) -> None:
        store_started = time.perf_counter()
        try:
            reserved = self._reserved_memory_bytes()
            if all(item is not active for item in self._active_requests):
                reserved = self._reserved_memory_bytes(
                    extra_capacity=active.cache.capacity
                )
            stored_blocks = 0
            if reserved <= self.generator.kv_budget_bytes:
                stored_blocks = self.generator.prefix_cache.store_completed_blocks(
                    active.request.input_ids,
                    active.cache,
                    reserved_memory_bytes=reserved,
                )
        except Exception:
            logger.exception(
                "prefix_cache_store_failed request_id=%s",
                active.request.request_id,
            )
            stored_blocks = 0
        finally:
            self.generator.decoder.release_cache(active.cache)
        store_seconds = time.perf_counter() - store_started
        result = GenerationResult(
            output_ids=active.output_ids,
            finish_reason=finish_reason,
            prefill_seconds=active.prefill_seconds,
            inter_token_seconds=active.inter_token_seconds,
            restore_seconds=active.restore_seconds,
            prefix_lookup_seconds=active.prefix_lookup_seconds,
            store_seconds=store_seconds,
            queue_seconds=queue_seconds,
            prefix=PrefixTrace(
                block_size=self.generator.prefix_cache.block_size,
                prompt_blocks=active.prompt_blocks,
                hit_tokens=active.hit_tokens,
                restored_tokens=active.restored_tokens,
                stored_blocks=stored_blocks,
            ),
        )
        if not active.job.future.done():
            active.job.future.set_result(result)
        self._finish_scheduled_request(result, queue_seconds, active.request.request_id)
        post_completion_reserved = self._reserved_memory_bytes(excluding=active)
        logger.info(
            "continuous_completed request_id=%s %s",
            active.request.request_id,
            self._memory_log_fields(post_completion_reserved),
        )

    def _fail_active_requests(self, error: Exception) -> None:
        for active in self._active_requests:
            self.generator.decoder.release_cache(active.cache)
            if not active.job.future.done():
                active.job.future.set_exception(error)
        self._active_requests = []
        self.generator.reserve_active_cache(0)

    def _reserved_memory_bytes(
        self,
        *,
        extra_capacity: int = 0,
        excluding: _ActiveRequest | None = None,
    ) -> int:
        active = [item for item in self._active_requests if item is not excluding]
        kv_bytes = sum(item.reservation_bytes for item in active)
        if extra_capacity:
            kv_bytes += self.generator.request_cache_bytes(extra_capacity)
        if self.generator.paged_attention:
            return kv_bytes
        count = len(active) + bool(extra_capacity)
        if count < 2:
            return kv_bytes
        capacities = [item.cache.capacity for item in active]
        if extra_capacity:
            capacities.append(extra_capacity)
        model_config = self.generator.decoder.model.config
        per_layer_bytes = self.generator.cache.bytes_per_token // model_config.n_layers
        max_capacity = max(capacities)
        batched_kv_bytes = count * max_capacity * per_layer_bytes
        attention_workspace_bytes = (
            count * model_config.n_heads * max_capacity * (1 + 4)
        )
        return kv_bytes + batched_kv_bytes + attention_workspace_bytes

    def _memory_log_fields(self, kv_reserved_bytes: int) -> str:
        cache = self.generator.cache
        prefix_cache_bytes = self.generator.prefix_cache.memory_bytes
        total_reserved_bytes = (
            cache.device_occupied_bytes
            + cache.activation_headroom_bytes
            + kv_reserved_bytes
            + prefix_cache_bytes
        )
        return (
            f"kv_reserved_bytes={kv_reserved_bytes} "
            f"prefix_cache_bytes={prefix_cache_bytes} "
            f"activation_headroom_bytes={cache.activation_headroom_bytes} "
            f"total_gpu_reserved_bytes={total_reserved_bytes}"
        )

    def _finish_scheduled_request(
        self, result: GenerationResult, queue_seconds: float, request_id: str
    ) -> GenerationResult:
        result = replace(result, queue_seconds=queue_seconds)
        generation_seconds = result.prefill_seconds + sum(result.inter_token_seconds)
        tokens_per_second = (
            len(result.output_ids) / generation_seconds
            if generation_seconds > 0
            else 0.0
        )
        logger.info(
            "request_completed request_id=%s finish_reason=%s output_tokens=%d "
            "cache_hit=%s cached_tokens=%d model_ttft_ms=%.1f "
            "generation_tok_s=%.2f total_ms=%.1f",
            request_id,
            result.finish_reason,
            len(result.output_ids),
            result.prefix.restored_tokens > 0,
            result.prefix.restored_tokens,
            (
                result.prefix_lookup_seconds
                + result.restore_seconds
                + result.prefill_seconds
            )
            * 1_000,
            tokens_per_second,
            (queue_seconds + generation_seconds) * 1_000,
        )
        return result

    def _validate_request(
        self, input_ids: list[int], eos_token_id: int, sampling: Sampling
    ) -> None:
        model_config = self.generator.decoder.model.config
        vocabulary_size = model_config.vocab_size
        if not input_ids:
            raise ValueError("A request prompt must contain at least one token.")
        token_ids = [eos_token_id, *input_ids]
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < vocabulary_size
            for token_id in token_ids
        ):
            raise ValueError(
                f"Token IDs must be between 0 and {vocabulary_size - 1:,}."
            )
        capacity = len(input_ids) + sampling.max_new_tokens
        context_length = model_config.context_length
        if capacity > context_length:
            raise ValueError(
                f"Request needs {capacity:,} cache positions, but the model supports "
                f"{context_length:,}."
            )
        max_tokens = (
            self.generator.kv_budget_bytes // self.generator.cache.bytes_per_token
        )
        if capacity > max_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled limit "
                f"is {max_tokens:,}."
            )
