"""Run with: uv run python -m unittest discover -s tests -v.

These CPU regressions intentionally fail where cohort draining prevents admission.
Only model loading and the scheduler thread are replaced; ticks, inference, KV
storage, prefix caching, and futures use the production implementations.
The synthetic memory budget does not validate CUDA peak memory or thread races.
"""

import unittest
from collections import Counter
from concurrent.futures import Future
from unittest.mock import Mock, patch

import torch

from helios.config import HeliosConfig
from helios.runtime.check import CacheCapacity, MemoryReport
from helios.runtime.engine import Engine
from helios.runtime.generate import GenerationResult
from helios.runtime.qwen3.config import Qwen3Config
from helios.runtime.qwen3.decode import Decoder
from helios.runtime.qwen3.loader import LoadedQwen3, Qwen3Loader
from helios.runtime.qwen3.model import Qwen3Model
from helios.runtime.types import Sampling


class ContinuousBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        config = Qwen3Config(
            vocab_size=64,
            context_length=8192,
            hidden_size=16,
            n_heads=2,
            n_layers=2,
            hidden_dim=32,
            head_dim=8,
            n_kv_heads=1,
            dtype=torch.float32,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = Qwen3Model(config).eval()
            with torch.no_grad():
                # Keep greedy outputs sensitive to attention and KV corruption.
                self.model.token_embedding.weight.mul_(0.1)
                self.model.token_embedding.weight[0].zero_()
        self.prefills: Counter[tuple[int, ...]] = Counter()
        self.decode_batch_sizes: list[int] = []

        def record_prefill(module, args):
            tokens = args[0]
            if tokens.shape[1] > 1:
                for row in tokens.tolist():
                    self.prefills[tuple(row)] += 1
            else:
                self.decode_batch_sizes.append(tokens.shape[0])

        hook = self.model.register_forward_pre_hook(record_prefill)
        self.addCleanup(hook.remove)

    def make_engine(self, *, slots: int = 8, budget_tokens: int = 8192) -> Engine:
        config = self.model.config
        element_size = self.model.token_embedding.weight.element_size()
        bytes_per_token = (
            config.n_layers * 2 * config.n_kv_heads * config.head_dim * element_size
        )
        budget_bytes = budget_tokens * bytes_per_token
        capacity = CacheCapacity(
            free_bytes=budget_bytes,
            device_occupied_bytes=0,
            activation_headroom_bytes=0,
            bytes_per_token=bytes_per_token,
            max_tokens=budget_tokens,
            kv_budget_bytes=budget_bytes,
            model_occupied_bytes=0,
        )
        report = MemoryReport(
            gpu="test-cpu",
            total_bytes=budget_bytes,
            max_gpu_utilization=1.0,
            max_gpu_bytes=budget_bytes,
            free_before_load_bytes=budget_bytes,
            weight_bytes=0,
            required_bytes=0,
            fits=True,
            cache=capacity,
        )
        loader = Mock(spec=Qwen3Loader)
        loader.load.return_value = LoadedQwen3(self.model, capacity, report, "test")
        with patch("helios.runtime.scheduler.Thread"):
            engine = Engine(
                HeliosConfig(
                    model_id="test-cpu",
                    hf_token=None,
                    max_batch_size=slots,
                    batch_wait_ms=0,
                ),
                loader=loader,
            )
        self.addCleanup(engine.close)
        return engine

    def enqueue(
        self,
        engine: Engine,
        name: str,
        prompt: list[int],
        output_limit: int,
        *,
        eos: int = 0,
    ) -> Future[GenerationResult]:
        return engine.enqueue(
            prompt,
            eos,
            Sampling(temperature=0, top_p=1, max_new_tokens=output_limit),
            request_id=name,
        )

    def tick(self, engine: Engine) -> None:
        engine._continuous_tick(engine._scheduler)

    def assert_active(self, engine: Engine, *names: str) -> None:
        self.assertCountEqual(engine.scheduler_snapshot()["active"], names)

    def finish(self, engine: Engine, *futures: Future[GenerationResult]) -> None:
        for _ in range(64):
            if all(future.done() for future in futures):
                for future in futures:
                    future.result(timeout=0)
                return
            self.tick(engine)
        self.fail("Requests did not complete within 64 scheduler iterations")

    def assert_matches_single(
        self, future: Future[GenerationResult], prompt: list[int], output_limit: int
    ) -> None:
        expected = Decoder(self.model).generate(
            prompt,
            0,
            Sampling(temperature=0, top_p=1, max_new_tokens=output_limit),
            max_total_tokens=self.model.config.context_length,
        )
        actual = future.result(timeout=0)
        self.assertEqual(actual.output_ids, expected.output_ids)
        self.assertEqual(actual.finish_reason, expected.finish_reason)
        self.assertEqual(len(actual.output_ids), output_limit)

    def test_observed_increasing_capacities_share_the_first_batch(self) -> None:
        engine = self.make_engine()
        futures = [
            self.enqueue(engine, name, [token] * length, 1024)
            for name, token, length in (
                ("decode_01", 3, 39),
                ("decode_02", 4, 43),
                ("decode_03", 5, 46),
            )
        ]
        self.tick(engine)

        self.assert_active(engine, "decode_01", "decode_02", "decode_03")
        self.assertEqual(engine.scheduler_snapshot()["waiting"], [])
        self.assertTrue(all(future.running() for future in futures))
        self.tick(engine)
        self.assertIn(3, self.decode_batch_sizes)

    def test_larger_late_arrival_joins_before_survivor_finishes(self) -> None:
        engine = self.make_engine()
        old_prompt, new_prompt = [3, 4, 5], [8, 9, 10, 11, 12, 13, 14]
        survivor = self.enqueue(engine, "survivor", old_prompt, 8)
        self.tick(engine)
        self.tick(engine)
        newcomer = self.enqueue(engine, "newcomer", new_prompt, 8)
        self.tick(engine)

        self.assertFalse(survivor.done())
        self.assert_active(engine, "survivor", "newcomer")
        self.assertTrue(newcomer.running())
        self.finish(engine, survivor, newcomer)
        self.assertIn(2, self.decode_batch_sizes)
        self.assertEqual(self.prefills[tuple(old_prompt)], 1)
        self.assert_matches_single(survivor, old_prompt, 8)
        self.assert_matches_single(newcomer, new_prompt, 8)

    def check_replacement(self, replacement_prompt: list[int]) -> None:
        engine = self.make_engine(slots=2)
        old_prompt, short_prompt = [3] * 9, [7, 8, 9] * 5
        short = self.enqueue(engine, "short", short_prompt, 2)
        survivor = self.enqueue(engine, "survivor", old_prompt, 8)
        self.tick(engine)
        self.assert_active(engine, "survivor", "short")
        replacement = self.enqueue(engine, "replacement", replacement_prompt, 8)
        self.tick(engine)

        self.assertTrue(short.done())
        self.assertFalse(survivor.done())
        self.assert_active(engine, "survivor", "replacement")
        self.finish(engine, survivor, short, replacement)
        self.assertIn(2, self.decode_batch_sizes)
        self.assertEqual(self.prefills[tuple(old_prompt)], 1)
        self.assert_matches_single(survivor, old_prompt, 8)
        self.assert_matches_single(short, short_prompt, 2)
        self.assert_matches_single(replacement, replacement_prompt, 8)

    def test_freed_slot_accepts_a_larger_request_without_draining(self) -> None:
        self.check_replacement([11] * 13)

    def test_freed_slot_reuse_preserves_surviving_and_replacement_outputs(self) -> None:
        self.check_replacement([11, 12, 13, 14, 15])

    def test_real_budget_shortage_waits_then_admits_after_release(self) -> None:
        engine = self.make_engine(slots=2, budget_tokens=22)
        prompt = [3, 4, 5, 6]
        first = self.enqueue(engine, "first", prompt, 10)
        second = self.enqueue(engine, "second", prompt, 10)
        self.tick(engine)
        self.assert_active(engine, "first")
        self.assertFalse(second.running())

        for _ in range(16):
            if first.done():
                break
            self.assert_active(engine, "first")
            self.assertEqual(engine.scheduler_snapshot()["waiting"], ["second"])
            self.tick(engine)
        self.assertTrue(first.done())
        self.assert_active(engine, "second")
        self.finish(engine, second)
        third = self.enqueue(engine, "third", prompt, 10)
        self.tick(engine)
        self.assert_active(engine, "third")
        self.finish(engine, third)
        for future in (first, second, third):
            self.assert_matches_single(future, prompt, 10)

    def test_mixed_lengths_and_prefix_restore_match_independent_generation(
        self,
    ) -> None:
        engine = self.make_engine(slots=2)
        prefix = list(range(1, 21))
        prime = self.enqueue(engine, "prime", prefix, 2)
        self.finish(engine, prime)
        extended = prefix + [21, 22]
        short = [30, 31, 32, 33, 34]
        cached = self.enqueue(engine, "cached", extended, 6)
        uncached = self.enqueue(engine, "uncached", short, 4)
        self.tick(engine)
        self.assert_active(engine, "cached", "uncached")
        self.finish(engine, cached, uncached)

        self.assertEqual(cached.result(timeout=0).prefix.restored_tokens, 16)
        self.assertEqual(uncached.result(timeout=0).prefix.restored_tokens, 0)
        self.assert_matches_single(cached, extended, 6)
        self.assert_matches_single(uncached, short, 4)

    def test_eos_releases_capacity_for_the_next_request(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=22)
        prompt = [3, 4, 5, 6]
        first_token = (
            Decoder(self.model)
            .generate(
                prompt,
                0,
                Sampling(temperature=0, top_p=1, max_new_tokens=1),
                max_total_tokens=22,
            )
            .output_ids[0]
        )
        stopped = self.enqueue(engine, "stopped", prompt, 10, eos=first_token)
        next_request = self.enqueue(engine, "next", [7, 8, 9, 10], 10)
        self.tick(engine)

        result = stopped.result(timeout=0)
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(result.output_ids, [])
        self.assert_active(engine, "next")
        self.finish(engine, next_request)
        self.assert_matches_single(next_request, [7, 8, 9, 10], 10)

    def test_cancelled_waiting_request_does_not_block_admission(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=22)
        first = self.enqueue(engine, "first", [3, 4, 5, 6], 10)
        cancelled = self.enqueue(engine, "cancelled", [7, 8, 9, 10], 10)
        last = self.enqueue(engine, "last", [11, 12, 13, 14], 10)
        self.tick(engine)
        self.assertTrue(cancelled.cancel())
        self.tick(engine)

        self.assertEqual(engine.scheduler_snapshot()["waiting"], ["last"])
        self.finish(engine, first, last)
        self.assertTrue(cancelled.cancelled())
        self.assert_matches_single(last, [11, 12, 13, 14], 10)

    def test_decode_failure_releases_capacity_for_waiting_request(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=22)
        failed = self.enqueue(engine, "failed", [3, 4, 5, 6], 10)
        next_request = self.enqueue(engine, "next", [7, 8, 9, 10], 10)
        self.tick(engine)
        forward = self.model.forward
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected decode failure")
            return forward(*args, **kwargs)

        with patch.object(self.model, "forward", side_effect=fail_once):
            self.tick(engine)
        with self.assertRaisesRegex(RuntimeError, "injected decode failure"):
            failed.result(timeout=0)
        self.assert_active(engine, "next")
        self.finish(engine, next_request)
        self.assert_matches_single(next_request, [7, 8, 9, 10], 10)


if __name__ == "__main__":
    unittest.main()
