import argparse
import json
import math
import os
import platform
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from suite import WARMUP_REQUEST

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
MESSAGE_ROLES = {"developer", "system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class BenchmarkRequest:
    request_id: str
    category: str
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    temperature: float
    top_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dataset of concurrent requests against an existing Helios server."
    )
    parser.add_argument("--label", default="run", help="Name for the saved result.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("HELIOS_BASE_URL", "http://127.0.0.1:8000"),
        help="URL of an already-running Helios server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1_800,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset.json",
        help="Versioned request dataset to run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum simultaneous HTTP requests (default: 8).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[int, list[BenchmarkRequest]]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Dataset does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Dataset is not valid JSON: {path}: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("schema_version"), int):
        raise TypeError("Dataset must be an object with an integer schema_version.")
    rows = data.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset requests must be a non-empty array.")

    requests: list[BenchmarkRequest] = []
    request_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Dataset request {index} must be an object.")
        request_id = row.get("id")
        category = row.get("category")
        messages = row.get("messages")
        max_tokens = row.get("max_tokens")
        temperature = row.get("temperature")
        top_p = row.get("top_p")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"Dataset request {index} needs a non-empty id.")
        if request_id in request_ids:
            raise ValueError(f"Dataset request id is duplicated: {request_id}")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Dataset request {request_id} needs a category.")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Dataset request {request_id} needs messages.")
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 2_048
        ):
            raise ValueError(f"Dataset request {request_id} has invalid max_tokens.")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(f"Dataset request {request_id} has invalid temperature.")
        if (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError(f"Dataset request {request_id} has invalid top_p.")

        normalized_messages: list[tuple[str, str]] = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in MESSAGE_ROLES
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError(
                    f"Dataset request {request_id} has an invalid message."
                )
            normalized_messages.append((message["role"], message["content"]))
        request_ids.add(request_id)
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                category=category,
                messages=tuple(normalized_messages),
                max_tokens=max_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
            )
        )
    return data["schema_version"], requests


def request_json(
    base_url: str,
    path: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"Helios returned HTTP {error.code} for {path}: {detail}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Cannot reach Helios at {base_url}. Start it first with `uv run helios`."
        ) from error


def run_request(
    base_url: str,
    model: str,
    spec: BenchmarkRequest,
    *,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        base_url,
        "/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {"role": role, "content": content}
                for role, content in spec.messages
            ],
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "max_tokens": spec.max_tokens,
            "stream": False,
        },
    )
    end_to_end_seconds = time.perf_counter() - started
    usage = response["usage"]
    timings = response["timings"]
    prompt_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]

    return {
        "id": spec.request_id,
        "category": spec.category,
        "input": spec.messages,
        "output": response["choices"][0]["message"]["content"],
        "timings": timings,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "end_to_end_seconds": end_to_end_seconds,
            "time_to_first_token_seconds": timings["time_to_first_token_seconds"],
            "generation_tokens_per_second": timings["generation_tokens_per_second"],
            "prefill_tokens_per_second": timings["prefill_tokens_per_second"],
            "decode_tokens_per_second": timings["decode_tokens_per_second"],
            "restored_tokens": usage["prompt_tokens_details"]["cached_tokens"],
            "cache_hit_rate": timings["cache_hit_rate"],
        },
    }


def run_continuous_batch(
    base_url: str,
    model: str,
    specs: list[BenchmarkRequest],
    *,
    timeout: float,
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(concurrency, len(specs))) as executor:
        samples = list(
            executor.map(
                lambda spec: run_request(base_url, model, spec, timeout=timeout), specs
            )
        )
    elapsed_seconds = time.perf_counter() - started
    total_output_tokens = sum(sample["metrics"]["output_tokens"] for sample in samples)
    return samples, {
        "request_count": len(specs),
        "concurrency": min(concurrency, len(specs)),
        "elapsed_seconds": elapsed_seconds,
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": total_output_tokens / elapsed_seconds,
    }


def duration(seconds: float) -> str:
    return f"{seconds * 1_000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def request_name(sample: dict[str, Any]) -> str:
    return f"{sample['category']}/{sample['id']}"


def print_response(index: int, total: int, sample: dict[str, Any]) -> None:
    print(
        f"\n[{index}/{total}] {request_name(sample)} response:\n"
        f"{sample['output']}",
        flush=True,
    )


def report(
    samples: list[dict[str, Any]],
    path: Path,
    warmup: dict[str, Any],
    continuous_batch_metrics: dict[str, Any],
) -> str:
    lines = [
        "",
        "Helios benchmark results",
        "",
        "Post-health warmup",
        f"End-to-end: {duration(warmup['metrics']['end_to_end_seconds'])}",
        f"Server total: {duration(warmup['timings']['total_seconds'])}",
        f"TTFT: {duration(warmup['timings']['time_to_first_token_seconds'])}",
        "",
        f"{'request':<28} {'in':>6} {'out':>6} {'E2E':>9} {'TTFT':>9} {'pre tok/s':>10} {'dec tok/s':>10} {'cache':>7}",
        "-" * 106,
    ]
    for sample in samples:
        metrics = sample["metrics"]
        name = request_name(sample)
        prefill_rate = metrics["prefill_tokens_per_second"]
        prefill_rate_display = (
            f"{prefill_rate:.2f}" if prefill_rate is not None else "—"
        )
        decode_rate = metrics["decode_tokens_per_second"]
        decode_rate_display = f"{decode_rate:.2f}" if decode_rate is not None else "—"
        ttft = metrics["time_to_first_token_seconds"]
        ttft_display = duration(ttft) if ttft is not None else "—"
        lines.append(
            f"{name:<28} {metrics['prompt_tokens']:>6} {metrics['output_tokens']:>6} "
            f"{duration(metrics['end_to_end_seconds']):>9} "
            f"{ttft_display:>9} "
            f"{prefill_rate_display:>10} {decode_rate_display:>10} "
            f"{metrics['cache_hit_rate'] * 100:>6.0f}%"
        )
    lines.extend(
        [
            "",
            "Continuous-batch metrics",
            f"Request count: {continuous_batch_metrics['request_count']}",
            f"Concurrent requests: {continuous_batch_metrics['concurrency']}",
            f"Output tokens: {continuous_batch_metrics['output_tokens']}",
            f"Elapsed: {duration(continuous_batch_metrics['elapsed_seconds'])}",
            (
                "Output throughput: "
                f"{continuous_batch_metrics['output_tokens_per_second']:.2f} tok/s"
            ),
        ]
    )
    lines.extend(["", f"Saved outputs and metrics: {path.relative_to(ROOT)}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    dataset_version, requests = load_dataset(args.dataset)
    print(f"Checking Helios at {args.base_url} ...", flush=True)
    health = request_json(args.base_url, "/health", timeout=args.timeout)
    model = health["model"]
    print(f"Model loaded: {model}", flush=True)
    print("Running post-health warmup ...", flush=True)
    warmup = run_request(
        args.base_url,
        model,
        BenchmarkRequest(
            request_id="warmup",
            category="warmup",
            messages=WARMUP_REQUEST.messages,
            max_tokens=WARMUP_REQUEST.workload.max_new_tokens,
            temperature=0.2,
            top_p=1.0,
        ),
        timeout=args.timeout,
    )
    print(
        f"Running {len(requests)} dataset requests at client concurrency "
        f"{min(args.concurrency, len(requests))} ...",
        flush=True,
    )
    samples, continuous_batch_metrics = run_continuous_batch(
        args.base_url,
        model,
        requests,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    for index, sample in enumerate(samples, start=1):
        print_response(index, len(samples), sample)

    now = datetime.now(UTC)
    record = {
        "schema_version": 5,
        "timestamp": now.isoformat(),
        "label": args.label,
        "machine": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "server": args.base_url,
        "accelerator": {"kind": "cuda", "name": health["memory"]["gpu"]},
        "model": {"id": health["model"], "revision": health["model_revision"]},
        "dataset": {
            "path": str(args.dataset),
            "schema_version": dataset_version,
            "request_count": len(requests),
        },
        "execution_mode": "continuous-batch",
        "continuous_batch_metrics": continuous_batch_metrics,
        "torch_compile": health["torch_compile"],
        "warmup": warmup,
        "samples": samples,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in args.label
    )
    path = RESULTS / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{safe_label}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(report(samples, path, warmup, continuous_batch_metrics))


if __name__ == "__main__":
    main()
