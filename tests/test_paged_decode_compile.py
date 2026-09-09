import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters

from helios.config import HeliosConfig
from helios.runtime.qwen3.config import Qwen3Config
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.qwen3.paged_cache import KVPagePool, PagedKVCache
from helios.runtime.types import Sampling
from helios.runtime.warmup import warm_decode


class PagedDecodeCompileTests(unittest.TestCase):
    def test_config_accepts_both_options(self):
        HeliosConfig(
            model_id="test", hf_token=None, torch_compile=True, paged_attention=True
        )

    def test_cpu_inductor_paged_decode(self):
        self.check_decode(torch.device("cpu"), torch.float32)

    def test_cuda_inductor_paged_decode(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("requires NVIDIA SM80 or newer")
        for dtype in (torch.float16, torch.bfloat16):
            self.check_decode(torch.device("cuda"), dtype)

    def check_decode(self, device, dtype):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=64,
            context_length=512,
            hidden_size=32,
            n_heads=2,
            n_layers=2,
            hidden_dim=64,
            head_dim=16,
            n_kv_heads=1,
            dtype=dtype,
        )
        model = Qwen3Model(config).to(device).eval()
        eager, compiled = Decoder(model), Decoder(model, torch_compile=True)
        for decoder in (eager, compiled):
            decoder.page_pool = KVPagePool(config, 16, device=device)
        sampling = Sampling(temperature=0, top_p=1, max_new_tokens=8)
        tolerance = 2e-5 if dtype == torch.float32 else 3e-2
        graph_counts = []
        for repeat in range(2):
            for lengths in ([1], [254, 7], [255, 8, 12]):
                caches = []
                for decoder in (eager, compiled):
                    with patch.object(
                        decoder,
                        "_compiled_paged_decode",
                        side_effect=AssertionError("prefill compiled"),
                    ):
                        caches.append(
                            [
                                decoder.prefill(
                                    [3] * length, sampling, max_total_tokens=512
                                ).cache
                                for length in lengths
                            ]
                        )
                for step in range(4):
                    expected = eager.decode_caches(caches[0], [4] * len(lengths))
                    with patch.object(
                        compiled,
                        "_compiled_paged_decode",
                        wraps=compiled._compiled_paged_decode,
                    ) as forward:
                        actual = compiled.decode_caches(caches[1], [4] * len(lengths))
                        self.assertEqual(forward.call_count, 1)
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                    for a, b in zip(*caches, strict=True):
                        self.assertEqual(a.length, b.length)
                        self.assertIsNone(b._pending_tokens)
                        for layer in range(config.n_layers):
                            for name in ("keys", "values"):

                                def gather(cache, name=name, layer=layer):
                                    pool = getattr(cache.pool, name)[layer]
                                    return torch.cat(
                                        [pool[p.index] for p in cache.pages]
                                    )[: cache.length]

                                torch.testing.assert_close(
                                    gather(a), gather(b), atol=tolerance, rtol=tolerance
                                )
                    if step == 1 and len(lengths) > 1:
                        for group in caches:
                            group.pop(0).close()
                        lengths = lengths[1:]
                for group in caches:
                    for cache in group:
                        cache.close()
                self.assertEqual(compiled.page_pool.free_pages, 16)
            graph_counts.append(counters["stats"]["unique_graphs"])
        self.assertEqual(graph_counts[0], graph_counts[1])

        restored = []
        for decoder in (eager, compiled):
            cache = decoder.prefill([3] * 256, sampling, max_total_tokens=512).cache
            snapshot = cache.snapshot_block(0, 256)
            cache.close()
            cache = PagedKVCache(decoder.page_pool, 264)
            cache.restore_blocks([snapshot])
            restored.append(cache)
        expected = eager.decode_caches([restored[0]], [4])
        actual = compiled.decode_caches([restored[1]], [4])
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for cache in restored:
            self.assertEqual(cache.length, 257)
            cache.close()
        del snapshot
        compiled.page_pool = KVPagePool(config, 3, device=device)
        _, sizes = warm_decode(
            compiled,
            [3] * 300,
            max_batch_size=2,
            max_tokens=512,
            budget_tokens=768,
        )
        self.assertEqual(sizes, (1, 2))
        self.assertEqual(compiled.page_pool.free_pages, 3)
