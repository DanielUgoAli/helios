import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters

from helios.runtime.qwen3.config import Qwen3Config
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling
from helios.runtime.warmup import DECODE_WARMUP_STEPS, warm_decode


class DecodeCompileTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = Qwen3Model(
                Qwen3Config(
                    vocab_size=64,
                    context_length=128,
                    hidden_size=16,
                    n_heads=2,
                    n_layers=1,
                    hidden_dim=32,
                    head_dim=8,
                    n_kv_heads=1,
                    dtype=torch.float32,
                )
            ).eval()
        self.sampling = Sampling(temperature=0, top_p=1, max_new_tokens=12)

    def test_prefill_never_calls_compiled_decode_even_for_one_token(self):
        decoder = Decoder(self.model, torch_compile=True)
        with patch.object(
            decoder, "_compiled_decode", side_effect=AssertionError("prefill compiled")
        ):
            for prompt in ([3], [3, 4, 5]):
                decoder.prefill(prompt, self.sampling, max_total_tokens=128)

    def test_inductor_logits_cache_mutation_and_graph_reuse(self):
        eager = Decoder(self.model)
        compiled = Decoder(self.model, torch_compile=True)
        graph_counts = []
        for repeat in range(2):
            for batch in (1, 2, 3, 8):
                prompts = [[3] * (5 + repeat + row * 2) for row in range(batch)]
                a = [
                    eager.prefill(p, self.sampling, max_total_tokens=128).cache
                    for p in prompts
                ]
                b = [
                    compiled.prefill(p, self.sampling, max_total_tokens=128).cache
                    for p in prompts
                ]
                for _ in range(4):
                    expected = eager.decode_caches(a, [4] * batch)
                    actual = compiled.decode_caches(b, [4] * batch)
                    torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
                for expected, actual in zip(a, b, strict=True):
                    self.assertEqual(expected.length, actual.length)
                    for x, y in zip(expected._layers, actual._layers, strict=True):
                        torch.testing.assert_close(x.keys, y.keys, atol=2e-5, rtol=2e-5)
                        torch.testing.assert_close(
                            x.values, y.values, atol=2e-5, rtol=2e-5
                        )
            graph_counts.append(counters["stats"]["unique_graphs"])
        self.assertEqual(graph_counts[0], graph_counts[1])
        a = eager.generate([3, 4, 5], 0, self.sampling, max_total_tokens=128)
        b = compiled.generate([3, 4, 5], 0, self.sampling, max_total_tokens=128)
        self.assertEqual(a.output_ids, b.output_ids)

    def test_warmup_fixed_steps_covers_every_batch_and_cleans_failure(self):
        decoder = Decoder(self.model)
        with patch.object(
            decoder, "decode_caches", wraps=decoder.decode_caches
        ) as decode:
            peak, sizes = warm_decode(
                decoder, [3] * 12, max_batch_size=3, max_tokens=128, budget_tokens=128
            )
            self.assertEqual(peak, 0)
            self.assertEqual(sizes, (1, 2, 3))
            counts = {size: 0 for size in sizes}
            for call in decode.call_args_list:
                counts[len(call.args[1])] += 1
            self.assertEqual(set(counts.values()), {4 * DECODE_WARMUP_STEPS})
        with patch.object(
            decoder, "decode_caches", side_effect=RuntimeError("injected")
        ), patch.object(
            decoder, "release_cache", wraps=decoder.release_cache
        ) as release:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                warm_decode(
                    decoder,
                    [3] * 12,
                    max_batch_size=3,
                    max_tokens=128,
                    budget_tokens=128,
                )
            self.assertEqual(release.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Reduce HELIOS_MAX_BATCH_SIZE"):
            warm_decode(
                decoder, [3] * 12, max_batch_size=3, max_tokens=128, budget_tokens=7
            )
