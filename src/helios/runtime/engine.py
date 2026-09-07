import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace
from threading import Lock

from helios.config import HeliosConfig
from helios.runtime.check import MemoryChecker
from helios.runtime.generate import GenerationResult, Generator, PrefixTrace
from helios.runtime.load import Loader
from helios.runtime.prefix_cache import PromptBlockView, describe_prompt_blocks
from helios.runtime.qwen3.cache import KVCache
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
    slot: int
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
        self.torch_compile = config.torch_compile
        self._memory_checker = MemoryChecker(config)
        self.report = loaded.report
        self.generator = Generator(
            loaded.model,
            loaded.cache,
            torch_compile=config.torch_compile,
            prefix_cache_ttl_seconds=config.prefix_cache_ttl_seconds,
        )
        self._generation_lock = Lock()
        self._max_batch_size = config.max_batch_size
        self._active_requests: list[_ActiveRequest] = []
        self._cohort_cache: KVCache | None = None
        self._cohort_capacity = 0
        self._cohort_slots = 0
        self._draining_cohort = False
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
            self._active_requests = []
            if self._cohort_cache is not None:
                self._release_cohort()

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
        """Advance active requests once, then fill any slots that became free."""
        with self._generation_lock:
            self._drop_cancelled_active()
            if self._active_requests:
                try:
                    self._decode_active_requests()
                except Exception as error:  # noqa: BLE001
                    self._fail_active_requests(error)

            scheduler.set_active(tuple(active.job for active in self._active_requests))

            if not self._active_requests and self._cohort_cache is not None:
                self._release_cohort()

            self._admit_requests(scheduler)
            scheduler.set_active(tuple(active.job for active in self._active_requests))
            return bool(self._active_requests or scheduler.peek() is not None)

    def _admit_requests(
        self, scheduler: Scheduler[_Request, GenerationResult]
    ) -> None:
        while not self._draining_cohort:
            head = scheduler.peek()
            if head is None:
                return
            request = head.payload
            needed = len(request.input_ids) + request.sampling.max_new_tokens
            if self._cohort_cache is None:
                try:
                    self._start_cohort(needed)
                except Exception as error:  # noqa: BLE001
                    job = scheduler.take(head)
                    if job is not None and not job.future.done():
                        job.future.set_exception(error)
                    continue
            if needed > self._cohort_capacity:
                self._draining_cohort = True
                return
            used_slots = {active.slot for active in self._active_requests}
            slot = next(
                (
                    candidate
                    for candidate in range(self._cohort_slots)
                    if candidate not in used_slots
                ),
                None,
            )
            if slot is None:
                return

            job = scheduler.take(head)
            if job is None:
                return
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
            try:
                lookup_started = time.perf_counter()
                prefix_hit = self.generator.prefix_cache.longest_prefix(
                    request.input_ids
                )
                prefix_lookup_seconds = time.perf_counter() - lookup_started
                prefill = self.generator.decoder.prefill_slot(
                    self._cohort_cache,
                    slot,
                    request.input_ids,
                    prefix_hit=prefix_hit,
                )
                active = _ActiveRequest(
                    job=job,
                    request=request,
                    slot=slot,
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
                    self.generator.decoder._sample(
                        prefill.logits, request.sampling
                    ).item()
                )
                if token_id == request.eos_token_id:
                    self._complete_request(active, "eos", queue_seconds)
                else:
                    active.output_ids.append(token_id)
                    if len(active.output_ids) == request.sampling.max_new_tokens:
                        self._complete_request(active, "length", queue_seconds)
                    else:
                        active.pending_token_id = token_id
                        self._active_requests.append(active)
            except Exception as error:  # noqa: BLE001
                if not job.future.done():
                    job.future.set_exception(error)
                self._cohort_cache.clear_slot(slot)
                continue

    def _start_cohort(self, capacity: int) -> None:
        max_tokens = self.generator.cache.max_tokens
        slots = min(self._max_batch_size, max_tokens // capacity)
        if slots < 1:
            raise RuntimeError("The FIFO request cannot fit in the KV-cache budget.")
        pool_bytes = slots * capacity * self.generator.cache.bytes_per_token
        self.generator.reserve_continuous_cache(pool_bytes)
        self._cohort_cache = self.generator.decoder.slot_cache(
            capacity=capacity, slots=slots
        )
        self._cohort_capacity = capacity
        self._cohort_slots = slots
        self._draining_cohort = False
        logger.info(
            "continuous_cohort_started slots=%d slot_capacity=%d kv_cache_tokens=%d",
            slots,
            capacity,
            slots * capacity,
        )

    def _decode_active_requests(self) -> None:
        if self._cohort_cache is None:
            raise RuntimeError("Active requests require a continuous KV-cache cohort.")
        slots = [active.slot for active in self._active_requests]
        tokens = [active.pending_token_id for active in self._active_requests]
        self.generator.decoder._synchronize()
        started = time.perf_counter()
        logits = self.generator.decoder.decode_slots(self._cohort_cache, slots, tokens)
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
            if token_id == active.request.eos_token_id:
                self._complete_request(
                    active, "eos", active.started_at - active.job.enqueued_at
                )
                continue
            active.output_ids.append(token_id)
            if len(active.output_ids) == active.request.sampling.max_new_tokens:
                self._complete_request(
                    active, "length", active.started_at - active.job.enqueued_at
                )
                continue
            active.pending_token_id = token_id
            surviving.append(active)
        self._active_requests = surviving

    def _drop_cancelled_active(self) -> None:
        if self._cohort_cache is None:
            return
        surviving: list[_ActiveRequest] = []
        for active in self._active_requests:
            if active.job.future.cancelled():
                self._cohort_cache.clear_slot(active.slot)
            else:
                surviving.append(active)
        self._active_requests = surviving

    def _complete_request(
        self, active: _ActiveRequest, finish_reason: str, queue_seconds: float
    ) -> None:
        if self._cohort_cache is None:
            raise RuntimeError("A completed request must still own its cache slot.")
        store_started = time.perf_counter()
        try:
            pool_bytes = (
                self._cohort_slots
                * self._cohort_capacity
                * self.generator.cache.bytes_per_token
            )
            stored_blocks = self.generator.prefix_cache.store_completed_blocks(
                active.request.input_ids,
                self._cohort_cache,
                reserved_memory_bytes=pool_bytes,
                slot=active.slot,
            )
        except Exception:
            logger.exception(
                "prefix_cache_store_failed request_id=%s",
                active.request.request_id,
            )
            stored_blocks = 0
        store_seconds = time.perf_counter() - store_started
        self._cohort_cache.clear_slot(active.slot)
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

    def _fail_active_requests(self, error: Exception) -> None:
        for active in self._active_requests:
            if not active.job.future.done():
                active.job.future.set_exception(error)
        self._active_requests = []
        self._release_cohort()

    def _release_cohort(self) -> None:
        self._cohort_cache = None
        self._cohort_capacity = 0
        self._cohort_slots = 0
        self._draining_cohort = False
        self.generator.reserve_continuous_cache(0)

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
        vocabulary_size = self.generator.decoder.model.config.vocab_size
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
        context_length = self.generator.decoder.model.config.context_length
        if capacity > context_length:
            raise ValueError(
                f"Request needs {capacity:,} cache positions, but the model supports "
                f"{context_length:,}."
            )
        max_tokens = (
            self.generator.cache.kv_budget_bytes // self.generator.cache.bytes_per_token
        )
        if capacity > max_tokens:
            raise ValueError(
                f"Request needs {capacity:,} KV-cache tokens, but the profiled limit "
                f"is {max_tokens:,}."
            )
