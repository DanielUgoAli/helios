from typing import TYPE_CHECKING

import torch

from helios.runtime.types import Sampling

if TYPE_CHECKING:
    from helios.runtime.qwen3.decode import Decoder


COMPILE_WARMUP_PROMPT = """
A regional library system has eight branches, a shared catalog, self-checkout kiosks, and a
mobile app. Patrons report that newly returned books sometimes remain unavailable for several
minutes, while staff occasionally see the same hold assigned twice during busy evenings. The
system uses one API, PostgreSQL, Redis, and a background worker. The team can make focused
changes but cannot replace these components. Recommend a staged reliability improvement that
includes data integrity, cache invalidation, observability, rollout safety, and user-facing error
handling. State the tradeoffs and define concrete metrics.
""".strip()

COMPILE_WARMUP_OUTPUT_TOKENS = 4


DECODE_WARMUP_STEPS = 4


def warm_decode(
    decoder: "Decoder",
    input_ids: list[int],
    *,
    max_batch_size: int,
    max_tokens: int,
    budget_tokens: int,
) -> tuple[int, tuple[int, ...]]:
    sampling = Sampling(temperature=0, top_p=1, max_new_tokens=DECODE_WARMUP_STEPS)
    device = decoder.device
    batch_sizes = tuple(range(1, max_batch_size + 1))
    activation_peak = 0
    for measured in (False, True):
        for batch_size in batch_sizes:
            capacity = min(max_tokens, budget_tokens // batch_size)
            prompt_limit = min(len(input_ids), capacity - DECODE_WARMUP_STEPS)
            if prompt_limit < 4:
                raise RuntimeError(
                    f"Decode warmup cannot fit batch size {batch_size} in the KV budget. "
                    "Reduce HELIOS_MAX_BATCH_SIZE."
                )
            for mixed in (False, True):
                caches = []
                result = None
                if measured and device.type == "cuda":
                    decoder._synchronize()
                    torch.cuda.empty_cache()
                    baseline = torch.cuda.memory_reserved(device)
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    for row in range(batch_size):
                        length = (
                            prompt_limit - int(measured) - (row % 2 if mixed else 0)
                        )
                        result = decoder.prefill(
                            input_ids[:length],
                            sampling,
                            max_total_tokens=max_tokens,
                        )
                        caches.append(result.cache)
                        result = None
                    for _ in range(DECODE_WARMUP_STEPS):
                        decoder.decode_caches(caches, [input_ids[-1]] * batch_size)
                    decoder._synchronize()
                    if measured and device.type == "cuda":
                        kv_bytes = sum(
                            cache.capacity * cache.memory_bytes_per_token
                            for cache in caches
                        )
                        peak = torch.cuda.max_memory_reserved(device) - baseline
                        activation_peak = max(activation_peak, peak - kv_bytes)
                finally:
                    result = None
                    for cache in caches:
                        decoder.release_cache(cache)
                    caches.clear()
                    cache = None
    return activation_peak, batch_sizes
