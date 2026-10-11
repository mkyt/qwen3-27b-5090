# Qwen3.8-27B on one RTX 5090

A single-purpose inference engine for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one NVIDIA RTX 5090
(32 GB, SM120), written from scratch in C++ / CUDA with a small Python server in front. It serves the full 262K-token
context, up to eight requests at once, images, and OpenAI- and Anthropic-compatible APIs, so coding agents such as
Claude Code and Codex can use it as their backend.

It is not a general engine: the kernels, the memory plan and the scheduler are written for this one model on this one
GPU, which is where its speed comes from.

| What | This engine | Notes |
| --- | ---: | --- |
| Decode, one request (code / JSON / prose) | 328–395 / 497–505 / 258–267 tok/s | DFlash2 speculative decoding, the model's default sampling |
| Aggregate decode, 4 / 8 concurrent requests | 935 / 1,395 tok/s | 4,096 output tokens each, about 172 tok/s per request at 8 |
| Prefill at 19K / 73K / 194K tokens | 11,295 / 8,773 / 5,875 tok/s | one request, cold |
| Agent turn after a 20K-token conversation | 0.19 s to the first token | prefix cache hit on everything reusable |
| KV cache capacity | 605,504 tokens | 262K context for one request, with `MAX_SEQS=8` |

On the same GPU with the same scripts, the fastest settings of other engines decode code at 285 tok/s (NInfer),
246 (SGLang with DFlash2) and 154 (vLLM) for one request, and reach 1,260–1,277 / 837–851 / 912–930 tok/s with eight
concurrent requests. A 152K-token prompt reaches its first token after 23 s here, 31 s on NInfer, 50 s on SGLang and
56 s on vLLM ([comparison](docs/benchmarks.md#other-engines)).

All numbers are at a 400 W power limit. [docs/benchmarks.md](docs/benchmarks.md) has the conditions, the commands that
produce them and the comparison with other engines.

## How

- **Weights**: [juzbox/Qwen3.8-27B-NVFP4-GDN](https://huggingface.co/juzbox/Qwen3.8-27B-NVFP4-GDN), NVIDIA's NVFP4
  checkpoint with the Gated DeltaNet projections also quantized to NVFP4 (GPTQ from the BF16 original). The attention
  projections stay FP8: quantizing them too cost parallel tool calls.
- **KV cache in NVFP4**, 1.7x the tokens of FP8, with the prefill attention's QK^T on native FP4 MMAs.
- **Speculative decoding** with the [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) block drafter, batched
  across concurrent requests; each request's tokens are bit-identical to running it alone.
- **Prefix cache** of GDN state checkpoints in pinned host memory, so an agent's growing conversation is prefilled once.
- **One memory plan at startup**: no allocation while serving; idle per-request areas are lent to the KV pool.
- **Vision tower** streamed from host memory layer by layer (no VRAM cost, 47 ms for a 1024x1024 image).

[docs/architecture.md](docs/architecture.md) explains the design; [docs/decisions.md](docs/decisions.md) lists the
measured reasons behind each choice, including the options that were tried and dropped. Task IDs in code comments
(such as `P7-8`) refer to entries there.

## Requirements

- An RTX 5090 (other GPUs are not supported: the kernels use SM120 block-scaled MMAs and are tuned for its 32 GB).
- Linux or WSL2 with a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.
- About 25 GB of disk for the weights and 20 GB for the build image. At least 16 GB of host RAM: the engine pins
  about 8 GB (the embedding table, the prefix cache, the vision tower).
- [uv](https://docs.astral.sh/uv/) on the host to download the weights.

## Quick start

```bash
scripts/download-weights.sh                 # gdnfp4 weights and the draft model into ./hf (23 GB)
docker compose up -d serve                  # builds the engine on first start, then serves 127.0.0.1:8000
docker compose logs -f serve                # wait for "engine ready"; the log shows the memory plan
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`./hf` can be a symlink if the weights should live on another disk.

## Clients

- **OpenAI-compatible** (chat completions, completions, Responses): base URL `http://127.0.0.1:8000/v1`, any API key,
  model `qwen3.8-27b`. Reasoning comes back in both `reasoning_content` and `reasoning`; `reasoning_effort` and
  `chat_template_kwargs.enable_thinking` are honored.
- **Anthropic-compatible** (messages, count_tokens), e.g. Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`, `ANTHROPIC_MODEL=qwen3.8-27b` (and the same for the small / fast model),
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`.
- **Codex** (Responses API): a provider with `base_url = "http://127.0.0.1:8000/v1"` and `wire_api = "responses"`.
- Send the previous turns' reasoning back: the chat template renders it, and the prefix cache only hits when it is there.
- Images as data URLs or http(s) URLs. No video.

Accepted but without effect: `response_format`, the penalties, `min_p`, `tool_choice` other than `auto` / `none`.
`n > 1` and `logprobs` are rejected.

## Configuration

The `serve` service reads its settings from the environment (or `.env`); the most useful ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_TOKENS` | 262144 | Context limit of one request |
| `MAX_SEQS` | 8 | Requests that can run at once (1–8); how many actually run is decided while running, from the free KV cache |
| `KV_DTYPE` | `nvfp4` | `nvfp4`, `fp8` or `bf16` |
| `WEIGHTS` | `gdnfp4` | `gdnfp4`, `nvidia` (NVIDIA's checkpoint as is), `quasar`, or a model directory |
| `SPEC` | `dflash2` | `dflash2` or `none` |
| `VISION` | `host` | `host`, `fp8`, `bf16` (vision tower placement) or `off` |

[docs/configuration.md](docs/configuration.md) lists every setting.

## Correctness and quality

The engine is checked against a PyTorch reference implementation of the model (`reference/`): the next-token
distributions over 35,328 positions of Japanese and English prose and code (the "golden" set, `bench/golden/`) agree
with the reference as closely as the reference agrees with itself under a different KV cache type (KL 0.036).
Against the BF16 original the default configuration is at KL 0.047; GSM8K, BFCL v3, long-context needle retrieval and
a Japanese set are in [docs/benchmarks.md](docs/benchmarks.md).

```bash
scripts/download-weights.sh --dev           # also NVIDIA's checkpoint, used by the reference model and the tests
scripts/download-golden.sh                  # the reference recordings (GitHub Releases)
docker compose --profile dev run --rm dev bash -c 'cd reference && python -m golden engine --prompts ../bench/golden/prompts.jsonl --out ../cache/engine.jsonl.gz'
docker compose --profile dev run --rm dev-cpu bash -c 'cd reference && python -m golden compare ../cache/engine.jsonl.gz ../bench/golden/ref-gpu.jsonl.gz'
```

`golden engine` loads the weights on the GPU (about 20 GB): stop the `serve` service first.

The recordings are about 8 MB each: the reference model with a BF16 KV cache (`ref-gpu`, what the engine is compared
with), with an FP8 KV cache (`ref-gpu-kvfp8`, the noise floor), the BF16 original (`ref-gpu-bf16model`) and images
(`vision-ref`). `reference/golden.py ref` records them again from the checkpoints in minutes (`ref-gpu-bf16model` needs the 55.6 GB BF16 original).

## Tests

The `dev` image builds the engine and runs the tests; `dev-cpu` is the same image without the GPU. Unit tests:

```bash
docker compose --profile dev run --rm dev-cpu python -m pytest -q frontend/tests                # server: parsing, admission (seconds)
docker compose --profile dev run --rm dev-cpu bash -c 'cd reference && pytest -q'               # reference model on the CPU (seconds; -m slow: every layer, about 11 min and 10 GB of RAM)
docker compose --profile dev run --rm dev bash -c 'cmake -S engine -B /build/engine -G Ninja && cmake --build /build/engine && cd engine/tests && python -m pytest -q'   # kernels against PyTorch (GPU, about 3 min)
```

End-to-end checks against a running server (`docker compose up -d serve`), from the host (standard library only):

```bash
python3 scripts/smoke_test.py                # a few requests
python3 scripts/api_test.py                  # OpenAI / Anthropic APIs: streaming, tool calls, reasoning
python3 scripts/vision_api_test.py           # images
```

The integration tests under `scripts/` (`decode_consistency.py`, `multiseq_test.py`, `prefix_cache_test.py`,
`needle_test.py`, `spec_sampling_test.py`, ...) each describe their use in the docstring at the top.

## Repository layout

```
engine/       the engine: CUDA kernels, model runner, speculative decoding, prefix cache, vision tower, C API
frontend/     the HTTP server (OpenAI / Anthropic APIs, chat template, tool-call parsing, scheduler)
reference/    PyTorch reference model, golden recordings
scripts/      weight download, end-to-end tests, benchmarks, quality evaluations, building gdnfp4
tools/        memory plan / roofline calculator, draft vocabulary builder
micro/        microbenchmarks of the GPU (bandwidth, MMA peaks, PCIe)
dev/          the build image (CUDA, CUTLASS, PyTorch for the reference)
bench/        golden prompts, draft vocabulary, test images
docs/         architecture, decisions, configuration, benchmarks
```

## License

MIT ([LICENSE](LICENSE)). The build image fetches [CUTLASS](https://github.com/NVIDIA/cutlass) (BSD-3-Clause); the
model weights (Qwen3.8-27B, NVIDIA's NVFP4 checkpoint, DFlash2, the gdnfp4 build) are Apache-2.0 and are downloaded,
not distributed here. The golden prompt texts are public domain or from this repository
([bench/golden/sources/README.md](bench/golden/sources/README.md)).
