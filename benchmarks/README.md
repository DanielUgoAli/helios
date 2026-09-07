# Dataset continuous-batching benchmark

Start Helios in one terminal. Model loading and the one-time compile warmup
belong to the server process:

```bash
HELIOS_TORCH_COMPILE=1 uv run helios
```

Then run the HTTP-only benchmark client in another terminal:

```bash
uv run python benchmarks/run.py --label continuous-batch
uv run python benchmarks/run.py --label continuous-batch --concurrency 16
uv run python benchmarks/run.py --label custom --dataset /path/to/dataset.json
```

The runner never loads, starts, stops, or owns the model. It calls the running
server's `/health` and `/v1/chat/completions` endpoints. It sends individual
chat-completion requests concurrently, so Helios's continuous scheduler can
share decode steps across in-flight requests. Use `--base-url` or
`HELIOS_BASE_URL` when the server is not listening on `http://127.0.0.1:8000`.

`dataset.json` is a versioned request file. Each entry supplies the same fields
used by the chat-completions API, except that the runner supplies the loaded
model ID and `stream: false`:

```json
{
  "schema_version": 1,
  "requests": [
    {
      "id": "decode_01",
      "category": "decode_heavy",
      "messages": [{"role": "user", "content": "..."}],
      "max_tokens": 1024,
      "temperature": 0.2,
      "top_p": 1.0
    }
  ]
}
```

After `/health`, the runner sends one isolated warmup request. It records its
client end-to-end latency, server total time, and TTFT separately; it is not a
dataset sample. Dataset samples retain their individual server timings, while
the record additionally contains continuous-batch aggregate elapsed time and
output throughput. The runner validates the dataset before contacting Helios.

Results are written to `benchmarks/results/`.
