# Qwen3.8-27B on one RTX 5090

A single-purpose inference engine for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one NVIDIA RTX 5090
(32 GB, SM120), written from scratch in C++ / CUDA with a small Python server in front. It serves OpenAI- and
Anthropic-compatible APIs with images, so coding agents such as Claude Code and Codex can use it as their backend.

It is not a general engine: the kernels, the memory plan and the scheduler are written for this one model on this one
GPU. That buys two things:

- **Speed.** Faster than [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang) and [NInfer](https://github.com/Neroued/ninfer) on the same card in every scenario measured: 1.3–2.8x the
  single-request decode, 1.1–1.7x the throughput of eight concurrent requests, and a 152K-token prompt answered in
  23 s instead of 31–56 s.
- **The full 262K context with room to spare.** A 605K-token KV cache on a 32 GB card, 2.8–6.3x what the other
  engines fit, with up to eight requests, images and speculative decoding all enabled at once. Of the others, only
  NInfer reached the model's whole 262,144-token context here, serving one request without speculative decoding.

All numbers are measured at a **400 W power limit** (the card's default is 575 W), on the same day, driver and scripts
for every engine. [docs/benchmarks.md](docs/benchmarks.md) has the conditions, each engine's settings and the
commands that reproduce them. At 575 W the engine is only 1–2% faster for one request and 9–26% faster for prefill
and eight concurrent requests, while 400 W gives 8–29% more tokens per joule
([Power limit and energy](docs/benchmarks.md#power-limit-and-energy)).

## Fast

| On one RTX 5090 at 400 W | This engine | NInfer | SGLang | vLLM |
| --- | ---: | ---: | ---: | ---: |
| Decode, one request: code / JSON / prose (tok/s) | **386 / 500 / 258** | 285 / 362 / 197 | 246 / 320 / 167 | 154 / 180 / 128 |
| Eight concurrent requests, total (tok/s) | **1,395** | 1,260–1,277 | 837–851 | 912–930 |
| Time to first token, 152K-token document | **23 s** | 31 s | 50 s | 56 s |
| Time to first token, agent turn after a 20K-token history | **0.19 s** | 0.39 s | 0.31 s | 0.57 s |
| Follow-up question on a 73K-token document | **0.14 s** | 10.9 s | 0.49 s | 1.01 s |

- **Speculative decoding** with the [DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) block drafter,
  batched across concurrent requests; each request's tokens are bit-identical to running it alone.
- **NVFP4 weights including the Gated DeltaNet projections**
  ([juzbox/Qwen3.8-27B-NVFP4-GDN](https://huggingface.co/juzbox/Qwen3.8-27B-NVFP4-GDN), GPTQ from the BF16 original;
  the attention projections stay FP8 because quantizing them cost parallel tool calls): decode reads 14 GiB of
  weights per step, and the GEMMs and attention run on FP4 / FP8 tensor-core MMAs.
- **Prefix cache** of the recurrent (GDN) state in host memory, so an agent's growing conversation and repeated
  questions on a long document are prefilled once.

## Long context on a 32 GB card

| | This engine | NInfer | SGLang | vLLM |
| --- | ---: | ---: | ---: | ---: |
| KV cache (tokens) | **605,504** | 96,128 | 214,219 | 173,913 |
| Longest context of one request | **262,144** (the model's limit) | 94,208 | 196,608 | 160,000 |
| ... at the same time as | 8 concurrent requests, images, DFlash2 | 4 requests | 4 requests, MTP | 8 requests, MTP |

The other engines' columns are the largest settings with concurrency and speculative decoding that ran on this
machine; NInfer and SGLang need other configurations for other scenarios (NInfer: 262K for one request without
speculative decoding; SGLang: one request at 90K for its DFlash2 mode). Here one configuration serves everything:
the defaults.

Where the memory comes from: the weights and the draft take about 15 GiB, and the engine plans the rest once at
startup against an explicit budget (no allocator slack, no profiling run). One GiB holds about 58K tokens of NVFP4
KV cache, so every GiB not spent elsewhere becomes context:

| Choice | VRAM it saves | About this many tokens of KV cache |
| --- | ---: | ---: |
| **KV cache in NVFP4** (4-bit values, an FP8 scale per 16), with prefill attention's QK^T on native FP4 MMAs: prefill is as fast as with FP8 KV or faster | 42% of the KV cache | 1.7x the tokens of FP8 (590K against 340K) |
| **Vision tower streamed from pinned host memory**, one layer ahead of the computation (47 ms for a 1024x1024 image, output identical to keeping it in VRAM) | 0.86 GiB of weights | 40K (measured against keeping it in VRAM) |
| **Embedding table in pinned host memory**, read over PCIe one row per token | 2.37 GiB | 138K |
| **Draft model quantized to NVFP4** at load time (same acceptance as BF16) | 2.4 GiB | 140K |
| **No per-row GDN state copies** for speculative verification (the next step replays the accepted tokens instead) | 0.9 GiB per request | 52K per request |
| **Idle requests' state lent to the KV cache**: the 189.6 MiB parking area of each waiting request holds KV pages until it is needed | 1.3 GiB with `MAX_SEQS=8` | 72K (measured) |

Long prompts stay fast and accurate:

| Context | Prefill | Decode, no speculation |
| ---: | ---: | ---: |
| 73K tokens | 8,773 tok/s (8.3 s) | 87 tok/s |
| 194K tokens | 5,875 tok/s (33 s) | 77 tok/s |

Between 73K and 152K tokens of a prompt this engine prefills 5,359 tok/s, against 3,868 (NInfer), 2,234 (SGLang)
and 1,953 (vLLM). Needle retrieval finds 16 of 16 facts at 190K tokens and 56 of 64 at 256K.

The full measurements, each engine's settings and the caveats (NInfer's memory sizing under WSL2, for one) are in
[docs/benchmarks.md](docs/benchmarks.md#other-engines); [docs/architecture.md](docs/architecture.md) explains the
design and [docs/decisions.md](docs/decisions.md) lists the measured reasons behind each choice, including the
options that were tried and dropped. Task IDs in code comments (such as `P7-8`) refer to entries there.

## Requirements

- An RTX 5090 (other GPUs are not supported: the kernels use SM120 block-scaled MMAs and are tuned for its 32 GB).
- Linux or WSL2 with a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.
- About 25 GB of disk for the weights and 350 MB for the serving image (the development image, for the tests and
  benchmarks, is 20 GB). At least 16 GB of host RAM: the engine pins about 8 GB (the embedding table, the prefix
  cache, the vision tower).
- [uv](https://docs.astral.sh/uv/) on the host to download the weights.

## Quick start

```bash
scripts/download-weights.sh                 # gdnfp4 weights and the draft model into ./hf (23 GB)
docker compose pull serve                   # the serving image from ghcr.io (350 MB)
docker compose up -d serve                  # serves 127.0.0.1:8000
docker compose logs -f serve                # wait for "engine ready"; the log shows the memory plan
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`./hf` can be a symlink if the weights should live on another disk.

To update: `git pull`, `docker compose pull serve`, `docker compose up -d serve`. Another build than the latest:
`SERVE_TAG=sha-<commit>` or a release version (the [package](https://github.com/mkyt/qwen3-27b-5090/pkgs/container/qwen3-27b-5090) lists them).

The serving image ([Dockerfile](Dockerfile), published by [a workflow](.github/workflows/image.yml)) compiles the
engine in the CUDA devel image and keeps only the result on plain Ubuntu: the CUDA runtime library (the driver comes
from the NVIDIA Container Toolkit), `libq27.so`, Python and the server's four packages (`frontend/requirements.txt`:
tokenizers, jinja2, numpy, pillow). No compiler, PyTorch or transformers. `docker compose up -d --build serve` builds it from the checkout instead (about 2 minutes the first
time, then only the engine sources that changed), e.g. after changing the code.

`scripts/serve.sh up` does the same and waits until the server answers (`up --idle`: the server only, no VRAM until
the engine is started at `/ui`; `up --build`; `update`: pull first; `down`, `status`). From Windows, double-click
`\\wsl.localhost\<distro>\<path to the checkout>\scripts\windows\serve.cmd`, or put a shortcut to it on the desktop
or in `shell:startup` (`serve.cmd idle` there, to start with the logon without taking the GPU). It starts the
container in the WSL distro the checkout is in, opens the stats page, and keeps the distro running while the
container runs (WSL shuts an idle distro down, and Docker with it).

http://127.0.0.1:8000/ui shows the server live: GPU utilization and power, VRAM by owner (weights, KV cache, state,
activations, CUDA runtime), each running request's context and prefill / decode speed, and a log of the requests
with their prefix-cache hits. The page also stops the engine to free the VRAM (the server stays up) and starts or
restarts it with other settings (weights, KV cache type, speculative decoding, vision tower, requests at once,
context limit, VRAM budget), showing for every choice the context it leaves before anything is loaded
([docs/configuration.md](docs/configuration.md#start-settings)).

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

The `dev` image (dev/Dockerfile: the CUDA toolkit, PyTorch) builds the engine and runs the tests; `dev-cpu` is the same image without the GPU. Every kernel is the engine's own, the prefill GEMMs included (`engine/src/gemm_mma.cu`, on
`mma.sync` and TMA; they replaced CUTLASS's SM120 kernels with bit-identical results), so the engine has no
dependency beyond the CUDA toolkit. Unit tests:

```bash
docker compose --profile dev run --rm dev-cpu python -m pytest -q frontend/tests                # server: parsing, admission (seconds)
docker compose --profile dev run --rm dev-cpu bash -c 'cd reference && pytest -q'               # reference model on the CPU (seconds; -m slow: every layer, about 11 min and 10 GB of RAM)
docker compose --profile dev run --rm dev bash -c 'cmake -S engine -B /build/engine -G Ninja && cmake --build /build/engine && cd engine/tests && python -m pytest -q'   # kernels against PyTorch (GPU, about 3 min)
```

End-to-end checks against a running server (`docker compose up -d serve`; `--build` to test a change), from the host (standard library only):

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
Dockerfile    the serving image (libq27.so, the server and four Python packages; 350 MB, ghcr.io/mkyt/qwen3-27b-5090)
dev/          the development image (CUDA, PyTorch for the reference and the tests)
bench/        golden prompts, draft vocabulary, test images
docs/         architecture, decisions, configuration, benchmarks
```

## License

MIT ([LICENSE](LICENSE)). The
model weights (Qwen3.8-27B, NVIDIA's NVFP4 checkpoint, DFlash2, the gdnfp4 build) are Apache-2.0 and are downloaded,
not distributed here. The golden prompt texts are public domain or from this repository
([bench/golden/sources/README.md](bench/golden/sources/README.md)).
