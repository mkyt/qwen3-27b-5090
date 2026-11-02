# Configuration

All settings are environment variables of `frontend/server.py`, with the same names and defaults however it is started
(the exceptions are in the first table below, and `HOST`). The startup log prints the memory plan. Task IDs such as `P7-13` refer to
[decisions.md](decisions.md).

## How settings work

**Keep a setting in one file; try one on the stats page.** A value in the file applies from the next start of the
server. A change on the stats page or through the admin API applies to the running server and is gone when it restarts.

| | Docker (`serve`, `serve.sh`, `serve.cmd`) | Windows package: `start.cmd` | Windows package: tray app |
| --- | --- | --- | --- |
| The file | `.env` next to `compose.yaml` | `settings.ini` next to the program | the same `settings.ini` |
| Changing the file | An editor | An editor | **Settings** in the tray menu ([below](#settings-window)), or an editor |
| Order | The shell's variables, then `.env`, then the default | `settings.ini`, then the environment, then the default | the same as `start.cmd` |
| Address and port | `HOST`, `PORT`: the address published on the host | `HOST`, `PORT` | `HOST`, `PORT` |
| Where the weights are | `./hf` (a symlink moves them; `HF_HOME` is `/hf` in the container) | `HF_HOME`, by default `huggingface\` in the folder | the same as `start.cmd` |
| Getting the weights | Before the start: `scripts/download-weights.sh` | Downloaded at each start if missing | the same as `start.cmd`, with the progress in the tray |
| `ENGINE_AUTOSTART` default | 1: loads at the start | 1: loads at the start | **0: loads at the first request** |
| `ENGINE_IDLE_UNLOAD_S` default | 0: stays loaded | 0: stays loaded | **1800: unloads after 30 minutes** |

What it takes for a setting to change depends on who reads it, and when:

| Takes effect | Settings | Change it for now | Keep it |
| --- | --- | --- | --- |
| At once | `ENGINE_IDLE_UNLOAD_S` | Stats page, `POST /admin/engine/idle` | The file |
| At once | `MIN_SEQ_TPS`, `ADMIT_KV`, `KV_RESERVE_TOKENS` | `POST /admin/concurrency` | The file |
| At once | `target_seqs` | `POST /admin/concurrency` | Cannot be kept: there is no variable (`MAX_SEQS` is the upper limit that can) |
| When the engine restarts | `WEIGHTS`, `KV_DTYPE`, `SPEC`, `VISION`, `MAX_SEQS`, `MAX_TOKENS`, `VRAM_BUDGET_GIB` | Stats page ([Start settings](#start-settings)), `POST /admin/engine` | The file. The page lists the lines to add, for these and for `ENGINE_IDLE_UNLOAD_S` |
| When the server restarts | Everything else, such as `HOST`, `PORT`, `HF_HOME`, `ENGINE_AUTOSTART`, `MIN_KV_TOKENS`, `MAX_TOKENS_FIT`, `STATE_HOST_CACHE`, `PREFILL_CHUNK` | Not possible | The file |

- Restarting the engine reloads the weights, fails the requests in flight and empties the prefix cache. Restarting the
  server does the same if the engine is loaded, and also closes the stats page's connection for a moment.
- `WEIGHTS` with another tokenizer takes a restart of the server ([Other checkpoints](#other-checkpoints)).
- With Docker, restart the server with `docker compose up -d serve` (or `scripts/serve.sh up`): it recreates the
  container when a value changed. `docker compose restart` keeps the old values. With `start.cmd`, close it and run it
  again. The tray app's settings window does what its changes need when you apply them.
- Only variables listed in the `serve` service of `compose.yaml` reach the container. That is all of this page except
  `HOST` and `PORT` (published by compose, above), the Windows package's, and most of
  [Development switches](#development-switches): add those in `compose.override.yaml`.

## Using it from other machines

By default Docker and the Windows package accept connections from this PC only. To use the server from other machines on
your network, set `HOST` to `0.0.0.0`:

| | What to do |
| --- | --- |
| Windows package, tray app | **Settings**, then **Address** `0.0.0.0` and **Apply**, or `HOST=0.0.0.0` in `settings.ini` |
| Windows package, `start.cmd` | `HOST=0.0.0.0` in `settings.ini`, then run it again |
| Docker on native Linux | `HOST=0.0.0.0` in `.env` next to `compose.yaml`, then `docker compose up -d serve`. Nothing else is needed |
| Docker under WSL2 | The same, and also one of the two settings below: WSL2 does not take connections from the network by default |

Use `0.0.0.0` rather than the PC's own address: the tray app and `scripts/serve.sh` check the server on `127.0.0.1`.

**Security:** there is no authentication. Anyone who can reach the port can use the model, and can stop the engine and
change its settings through the admin API and the stats page. Do this only on a network you trust, and do not forward the
port to the internet.

**Windows Firewall:** the first start of the Windows package with `0.0.0.0` asks whether to allow Python; allow it for
private networks. If you closed that prompt, add a rule in an administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Qwen3.8-27B" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

**Linux firewalls:** Docker publishes ports with its own iptables rules, ahead of `ufw` and `firewalld`'s usual rules, so
the port opens without a firewall rule and those tools' rules do not close it. To limit who can connect, use Docker's
`DOCKER-USER` chain or a firewall in front of the PC.

**WSL2** ([Microsoft's documentation](https://learn.microsoft.com/en-us/windows/wsl/networking)), either of:

- Mirrored networking (Windows 11 22H2 or later): add `networkingMode=mirrored` under `[wsl2]` in `%USERPROFILE%\.wslconfig`,
  run `wsl --shutdown`, and allow the port in an administrator PowerShell:

  ```powershell
  New-NetFirewallHyperVRule -Name "Qwen3.8-27B" -DisplayName "Qwen3.8-27B" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8000
  ```

- A port proxy, in an administrator PowerShell, plus the Windows Firewall rule above. The WSL address can change when WSL
  restarts; run it again then.

  ```powershell
  netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=(wsl hostname -I).Trim()
  ```

**On the other machine,** use the PC's address in place of `127.0.0.1`: the API at `http://<address>:8000/v1` (Anthropic
clients: `http://<address>:8000`) and the stats page at `http://<address>:8000/ui`. With another `PORT`, change the port
everywhere above.

## Serving

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` | `frontend/server.py` run directly: 0.0.0.0. Docker and the Windows package: 127.0.0.1 | Address to listen on. `0.0.0.0`: the network ([Using it from other machines](#using-it-from-other-machines)). With Docker, the address published on the host: the server in the container always listens on all its addresses |
| `PORT` | 8000 | Port of the API and the web UI. With Docker, the port published on the host; in the container it stays 8000 |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Name in `/v1/models` (requests may use any model name) |
| `MAX_TOKENS` | 262144 | Context limit of one request. If the KV pool cannot hold it, `MAX_TOKENS_FIT` lowers it; the value in effect is in the startup log (`max_tokens ... lowered to`) and in `/v1/models` (`max_model_len`) |
| `MAX_TOKENS_FIT` | 1 | 1: start with `MAX_TOKENS` lowered to what the KV pool holds (a multiple of 1,024). 0: fail to start instead (the service then restarts every 60 s). The engine library (`q27lib.Engine`) defaults to 0 |
| `MAX_SEQS` | 8 | Requests the engine can run at once (1–8); more wait in a queue. How many actually run is decided while running (`ADMIT_KV`, `MIN_SEQ_TPS`, `target_seqs` of the admin API). Each request beyond the first has a 189.6 MiB parking area at the top of the KV pool, lent to the KV cache while unused (P7-13), so `MAX_SEQS=8` costs about 0.25 GiB of fixed per-request buffers. KV pool with gdnfp4 and NVFP4 KV: 620,096 tokens with 4, 605,504 with 8 |
| `ADMIT_KV` | 1 | 1: a waiting request starts only when the KV pool can take it: free pages plus pages held only by prefix-cache checkpoints must cover the remaining prompts of the running requests, `KV_RESERVE_TOKENS` of output per request, and one more parking area. With nothing running a request always starts. 0: start by count only (a request that later runs out of KV is suspended) |
| `KV_RESERVE_TOKENS` | 4096 | Output tokens per request that `ADMIT_KV` keeps room for (or the request's remaining `max_tokens`, if smaller) |
| `MIN_SEQ_TPS` | 0 | Lower bound of tokens per second per request; 0 disables it. Another request does not start if it would push each below this. The prediction uses the measured speed at each number of running requests (decode turns, about the last 10 s); unmeasured counts are scaled from smaller ones by the step time |
| `DECODE_SHARE` | 0.5 | Seconds of decoding given to running requests per second of prefill, so output does not stall during a long prefill |
| `SCHED_QUANTUM_MS` | 50 | Length of a decode turn (with `BATCH_DECODE=0`, the interval between requests' turns; a switch costs about 1 ms) |
| `BATCH_DECODE` | 1 | 1: requests in decode advance together in one batched speculative step (P6-10; about 2.3x aggregate at 4 requests; each request's tokens are the same as alone). 0: requests take turns |
| `PREFILL_CHUNK` | 2048 | Prefill chunk in tokens (8,192 is only 2.5% faster at 32K and needs 4x the activations, P5-4) |
| `KV_DTYPE` | `nvfp4` | `nvfp4`, `fp8` or `bf16`. NVFP4 holds about 1.7x the tokens of FP8 (589,888 against 339,968 at 262K with gdnfp4), prefills as fast or faster (+1 / +6 / +9% at 19K / 73K / 194K, P7-8), and costs a little KL against the BF16 original (0.0470 against 0.0445 with FP8 KV); GSM8K, the Japanese set and needle retrieval are the same. BFCL `parallel` is about 3 points lower than FP8 on short prompts and the same after 32K / 96K of agent context (P4-11, P7-8, P7-9) |
| `WEIGHTS` | `gdnfp4` | `gdnfp4`: GDN projections in NVFP4, attention projections FP8 ([juzbox/Qwen3.8-27B-NVFP4-GDN](https://huggingface.co/juzbox/Qwen3.8-27B-NVFP4-GDN), pinned revision, `scripts/download-weights.sh`; PQ-14). `nvidia`: NVIDIA's checkpoint as is (GDN / attention projections FP8). `quasar`: the QUASAR-QAT checkpoint, every projection NVFP4, weaker at parallel tool calls (`scripts/download-weights.sh --quasar`; its BF16 `lm_head` is replaced by NVIDIA's NVFP4 one through `Q27_LM_HEAD_CKPT`). Or another NVFP4 checkpoint of Qwen3.8-27B on the Hugging Face Hub, e.g. a fine-tune: `ORG/NAME` (its main branch) or `ORG/NAME@REVISION`, downloaded with `scripts/download_weights.py --weights ORG/NAME` (which first checks that the engine takes it) or `scripts/download-weights.sh --repo ORG/NAME` (see [Other checkpoints](#other-checkpoints)). Or a model directory inside the container. From Python: `qwen27b.weights.engine_target(...)`; scripts take `--weights` |
| `SPEC` | `dflash2` | `dflash2` (speculative decoding with the DFlash2 draft) or `none` |
| `Q27_DRAFT_QUANT` | `fp4` | Draft linear layers: `fp4`, `fp8` or `bf16` (same acceptance, FP4 is fastest) |
| `Q27_DRAFT_VOCAB` | `bench/draft-vocab/vocab-128k.txt` | Candidate vocabulary of the draft (`tools/draft_vocab.py`); `none` for the full vocabulary |
| `STATE_HOST_CACHE` | 4 | GiB of pinned host memory for prefix-cache checkpoints (4 = 21 checkpoints; 0 disables the cache; the server sets 4 when unset, the engine library defaults to 0). The slots are not allocated at startup (pinning 4 GiB takes about 1.8 s on WSL2): one by one while the server is idle, or when a checkpoint needs one |
| `KV_POOL_TOKENS` | 0 | KV pool size in tokens (0 = all the VRAM budget leaves). When set, it is the KV cache with every parking area in use; the pool is that plus the parking areas (P7-13) |
| `CHECKPOINT_INTERVAL` | 8192 | Distance between checkpoints inside a long prompt (a multiple of 64) |
| `VISION` | `host` | Image input. `host`: the vision tower's BF16 weights (0.86 GiB) stay in pinned host memory and are streamed to the GPU one layer ahead of the computation; the KV pool loses 192 tokens, small images take +13 ms (the copy is about 20 ms), 1024x1024 and larger +1–4 ms; output identical to `bf16`. `fp8`: E4M3 weights with per-row scales on the GPU, widened per layer (0.43 GiB; embedding error about 3x that of `bf16`, answers can change). `bf16`: BF16 weights on the GPU (fastest, 0.86 GiB). `off`: requests with images get HTTP 400 (the engine library's default). Activations and the weight staging (90 MiB) share the prefill's buffers |
| `IMAGE_MAX_PIXELS` | 4194304 | Images are resized to at most this many pixels (2048x2048 = 4,096 tokens, 370 ms in the vision tower) |
| `IMAGE_TOKENS` | 16384 | Image tokens per request (embeddings in pinned host memory, 10 KiB per token: 160 MiB) |
| `VRAM_BUDGET_GIB` / `WDDM_MARGIN_GIB` | free − 0.6 / 0.6 | VRAM budget and the margin kept free for the display driver. The budget is never above the automatic one (free − margin, at most 30.3 GiB for the process): beyond it the driver moves memory to system RAM without an error, about 7 times slower. A higher `VRAM_BUDGET_GIB` is lowered, with a line in the log |
| `Q27_DEVICE_FREE_BYTES` | set by the server | Free VRAM with other programs counted (NVML), which the engine's budget uses if it is below what `cudaMemGetInfo` says |
| `MIN_KV_TOKENS` | 0 | Do not start the engine if the KV pool would hold fewer tokens (other programs are using the GPU, or a low budget). 0: start with the KV pool that fits ([Too little VRAM](#too-little-vram)) |
| `Q27_LOAD_THREADS` | 8 | Threads that read the weight files ahead of the uploads (the bytes loaded are the same with any number). On WSL2 one read stream from a Windows drive (`/mnt/d`, virtiofs) gets 1.2–1.5 GB/s; 4 to 16 threads reach the filesystem's 2.6–2.8 GB/s, and the engine starts in about 10 s instead of 18–21 s. A copy of the weights on the Linux file system, `cache/weights/<org>--<name>/<revision>/` (`cp -L` of the HF snapshot; used before the HF cache when present; the serve container sees it when `compose.override.yaml` mounts `./cache/weights:/work/cache/weights:ro`), starts in about 3.3 s while it is in the page cache |
| `SERVE_TAG` | `latest` | Tag of the serving image `ghcr.io/mkyt/qwen3-27b-5090` (compose): `latest` (main), a version such as `1.0.0`, or `sha-<commit>`. `docker compose pull serve` fetches it; `--build` builds it from the checkout instead |
| `ENGINE_AUTOSTART` | 1 | 0: the server starts with the engine stopped (no VRAM in use). Requests get 503 until it is started at `/ui` or with `POST /admin/engine`; with `ENGINE_IDLE_UNLOAD_S` above 0, the first generation request loads it instead |
| `ENGINE_IDLE_UNLOAD_S` | 0 | Seconds without a request after which the engine stops and frees its VRAM; the next generation request loads it again and waits for it ([Unloading when idle](#unloading-when-idle)). 0: off, the engine stays ready |
| `STATS_SAMPLE_S` | 1 | Seconds between the GPU samples (NVML: utilization, power, clocks, temperature, memory) of the stats page `/ui`; the last 10 minutes are kept. 0: no sampling (the page still shows requests, VRAM and the log) |
| `API_DUMP_DIR` | unset | Debugging: write the body of every POST to this directory, created if missing (to see what a client sends; may contain personal data). In the Windows package, a full path in `settings.ini` |

### Attention numerics

| Variable | Default | Meaning |
| --- | --- | --- |
| `Q27_ATTN_FP8` | empty | Prefill attention MMAs, read per call. Empty: `2` with FP8 KV, `3` with NVFP4 KV (`2` without the staging buffer). FP8 KV: `2` QK^T and P·V in FP8, `1` P·V only, `0` BF16. NVFP4 KV: `3` QK^T on native FP4 MMAs (Q quantized to NVFP4, K as cached, V from the E4M3 staging buffer; P7-8), `2` K and V rounded to E4M3 for FP8 MMAs, otherwise exact in f16 |
| `Q27_ATTN_FP4_STAGE` | 1 | NVFP4 KV prefill: `1` converts the context to an E4M3 staging buffer once per layer and chunk (mode 3: V only, 1 KiB per token of `max_tokens`, 256 MiB at 262K; with `Q27_ATTN_FP8=2` at startup K and V, 2 KiB per token, bit-identical to per-CTA conversion). `0`: convert in every CTA (mode 2). Sized at startup, read per call |
| `Q27_ATTN_TC_F8` | empty | Decode / verify attention MMAs, read per call. Empty: FP8 with NVFP4 KV (P7-7), `1`: FP8 also with FP8 KV, `0`: BF16 / f16. FP8 means Q in E4M3 with per-row scales, NVFP4 KV re-rounded to E4M3, P in E4M3: 8% faster speculative steps at 131K and at 8 x 32K, no change on golden or needle retrieval |

## Concurrency

- Up to `MAX_SEQS` requests run at once. Prefill goes to the request with the least prompt left; requests in decode
  advance together in batched steps (with `BATCH_DECODE=0` they take turns every `SCHED_QUANTUM_MS`).
- New requests wait until the KV pool can take them (`ADMIT_KV`). If the KV cache still runs out, the latest request
  (including the one that ran out) is suspended to a checkpoint and goes back to the head of the queue. When requests
  that arrive together share a prefix, the later one waits until the earlier one has passed it.
- Admin API (no authentication; Docker and the Windows package listen on 127.0.0.1 unless `HOST` says otherwise): `GET /admin/concurrency` returns the settings
  and the state (running, waiting, why requests wait, measured tok/s per number of running requests);
  `POST /admin/concurrency` changes `target_seqs` (1 to `MAX_SEQS`; lowering it does not stop running requests),
  `min_seq_tps`, `admit_kv` and `kv_reserve`. Changes survive an engine restart, not a service restart.
  Example: `curl -s -X POST localhost:8000/admin/concurrency -d '{"target_seqs": 4}'`.
- `/metrics` (Prometheus): among others `q27:target_seqs`, `q27:admission_blocked{reason=...}`,
  `q27:admission_waits_total{reason=...}`, `q27:cached_tokens_total`, `q27:reusable_tokens_total`.

## Stats page

`http://localhost:8000/ui` shows the server live in the browser (no authentication, like the admin API):

- GPU utilization, power against the limit, clocks and temperature (NVML), and the aggregate decode and prefill
  tokens per second, over the last 1 to 10 minutes.
- Device memory by owner: weights (by part of the model), the KV pool (pages of running requests, pages only
  prefix-cache checkpoints hold, parked sequence state, free), sequence state, activations, the CUDA context and
  runtime (CUDA graphs, libraries), other programs and the driver, and free memory, plus the pinned host pools. The
  pool sizes come from the engine's memory plan. The bar always shows the whole device from NVML, updated every
  second: without an engine, other programs and free memory; with one, what other programs take or release after it
  started is the change of the memory in use beyond the engine's own allocations (`cudaMemGetInfo`).
- Each running or waiting request: its context (tokens restored from the prefix cache, prefilled, still to run,
  generated, with the message boundaries), current and average prefill and decode tokens per second, tokens per
  speculative step, TTFT, time queued.
- A log of the finished requests (prompt, cached tokens and hit rate, output, TTFT, prefill / decode tok/s, finish
  reason, the start of the last user message) and of engine events (admission waits, preemptions, errors,
  restarts), the last 500 entries.

The page also starts, stops and restarts the engine (a start takes about 3.3 s with the weights in the page cache, about 10 s
reading them from a Windows drive under WSL2: see `Q27_LOAD_THREADS`). The engine process holds all the VRAM; the HTTP server
holds none, so stopping the engine frees the GPU for other work while the service and the page stay up. Requests in
flight fail with 503, and so do new ones until the engine is started again. The prefix cache is lost. The same from
a script: `curl -s -X POST localhost:8000/admin/engine -H 'Content-Type: application/json' -d '{"action": "stop"}'`
(`start`, `stop` or `restart`; `GET /admin/engine` returns the state: `stopped`, `loading`, `ready`, `restarting`
after a crash, `stopping`). The JSON content type is required, so that another web page cannot send the action as a
form.

### Unloading when idle

With `ENGINE_IDLE_UNLOAD_S` above 0 the engine stops after that many seconds without a generation request, so that the
GPU is free for other work, and the next generation request starts it and waits for it instead of getting 503. It is off by
default: the first request after an unload takes the load time longer (see above), and the prefix cache is lost, so
that request and the next ones with its context prefill from the start. Polling `/health`, `/metrics`, `/v1/models` or the stats page does not count as
a request and does not keep the engine loaded.

- `/health` answers 200 with `{"status": "unloaded"}` while unloaded (with `refused`, the reason, after a load that found
  too little VRAM) and `{"status": "loading"}` during such a load: the service can take requests.
  `GET /admin/engine` has `state: "stopped"`, `unloaded: true` and `idle_unload_s`.
- The memory plan is made when the engine loads, from the VRAM free then ([Too little VRAM](#too-little-vram)).
- A request waits at most 5 minutes for the load, then gets 503. Clients with a shorter timeout see their own timeout.
- `ENGINE_AUTOSTART=0` starts the server in this unloaded state: no VRAM in use until the first request.
- A stop from the stats page or `POST /admin/engine` stays until a start from there: requests get 503, as without this
  setting.
- The stats page tells these states apart (`engine unloaded`: the first request loads it; `engine stopped`: requests get
  503 until a start), shows the time left before an unload and why a load was refused, and changes the time under
  **Unload when idle** in the start settings. From a script:
  `curl -s -X POST localhost:8000/admin/engine/idle -H 'Content-Type: application/json' -d '{"idle_unload_s": 600}'`
  (0: off). It applies at once, the time running from then, and lasts until the service restarts.

### Too little VRAM

The engine plans its memory once, when it loads, from the VRAM free at that moment, with other programs counted (NVML; CUDA
alone does not see them, on WSL2 or on Windows), and allocates nothing beyond that budget. NVML catches up with a program's
allocation within about 2 s, so a load in that moment can count too little. If other programs are using the GPU then, what
happens depends on the settings:

| Setting | Effect |
| --- | --- |
| Default | The KV pool takes what is left. If it cannot hold `MAX_TOKENS`, `MAX_TOKENS` is lowered as well (`MAX_TOKENS_FIT=1`) |
| `MIN_KV_TOKENS=N` | The engine does not start if the KV pool would hold fewer than N tokens (for scale: about 610K with nothing else on the GPU, nvfp4) |
| `MAX_TOKENS_FIT=0` | The engine does not start if the KV pool cannot hold `MAX_TOKENS` |

For scale, with the defaults (gdnfp4, nvfp4, `MAX_SEQS=8`), by the memory other programs use: nothing, a KV pool of about 610K
tokens; 6 GiB, about 230K and `MAX_TOKENS` lowered to about 230K; 8 GiB, 120K; from about 9 GiB the engine does not start, unless `MAX_SEQS` is lower.

An engine that does not start allocates no pools. With `ENGINE_IDLE_UNLOAD_S` set, the requests waiting for it get 503 with the
reason (also `refused` in `GET /admin/engine`), and the next request tries again; there are no retries in between. Without it,
the server retries as after a crash, up to every 60 s.

**This covers the load only, by design.** The engine does not watch the GPU afterwards. A program that starts allocating VRAM
while the engine is loaded can push memory to system RAM, its own or the engine's, without an error. Before starting other GPU
work, unload the engine (the stats page, `POST /admin/engine`, or `ENGINE_IDLE_UNLOAD_S`).

`GET /ui/stream` is the page's event stream (server-sent events) and `GET /ui/state` returns one snapshot as JSON.

### Start settings

The page's header shows the running engine's start settings (with the engine stopped, the next start's); clicking
them opens "Start settings", which choose what the next start or restart of the engine uses, in place of these
variables:

| Setting | Variable | Values |
| --- | --- | --- |
| `weights` | `WEIGHTS` | `gdnfp4`, `nvidia`, `quasar` (those downloaded), or the model directory the service started with |
| `kv_dtype` | `KV_DTYPE` | `nvfp4`, `fp8`, `bf16` |
| `spec` | `SPEC` | `dflash2`, `none` |
| `vision` | `VISION` | `host`, `fp8`, `bf16`, `off` |
| `max_seqs` | `MAX_SEQS` | 1 to 8 |
| `max_tokens` | `MAX_TOKENS` | 2,048 to 262,144 (lowered to what the budget holds, `MAX_TOKENS_FIT`) |
| `vram_budget_gib` | `VRAM_BUDGET_GIB` | `null` (automatic) or 4 GiB up to the automatic budget |

Under every choice the page shows the longest context of one request and the KV pool all requests share if only that
setting changed, and for the settings as chosen the device pools against the budget. These come from the engine's own
memory planner (`q27_plan`: the pools of an engine start, nothing allocated or loaded, about 10 to 80 ms each), so
they are the numbers the startup log then prints. The free device memory they assume is the running engine's (before
its pools; a restart sees the same), or with the engine stopped the free memory NVML reports now less an engine's
CUDA context (about 1.2 GiB on WSL2, measured at each start).

The settings stay for later starts and restarts (also after a crash) until the service restarts, which starts from
the variables again: to keep them, set the variables, which `GET /admin/engine` lists in `keep_env` and the page shows
(in `settings.ini` for the Windows package). Changes of `/admin/concurrency` stay too (`target_seqs` only while `max_seqs` is the same).
The same from a script:

- `GET /admin/engine`: the state, `running` (the running engine's settings), `settings` (the next start's),
  `at_start` (those the service started with), `keep_env`,
  `weights` (null for each profile that can be loaded, else why not) and `device` (the free and total memory the
  plans assume).
- `POST /admin/engine/plan` with `{"settings": {...}}` (changes to the next start's settings): `plan` (`max_tokens`,
  `kv_tokens`, `budget`, `device` and `pools` in bytes, `fits`, or `error`), `variants` (the plans with each
  setting's other values), `weights` and `at_start`. `frontend/plan.py` answers the same without a server.
- `POST /admin/engine` with `{"action": "start" | "restart", "settings": {...}}`: starts with those changes, if they
  fit (HTTP 400 otherwise; 409 for `start` while the engine runs). With `"action": "save"` (the page's **Save for the
  next load**) they only stay for the next load, while the engine is stopped or unloaded (409 otherwise): it does not
  load, and whether they fit is decided when it does. Example:
  `curl -s -X POST localhost:8000/admin/engine -H 'Content-Type: application/json' -d '{"action": "restart", "settings": {"kv_dtype": "fp8", "max_seqs": 4}}'`.

## Prefix cache

The server takes checkpoints at message boundaries and at the end of each generation; the next request resumes from
the deepest match. The log line `[api] prompt N tokens, cached C of R reusable` and the `q27:cached_tokens_total` /
`q27:reusable_tokens_total` metrics show how well it hits. Clients must send the previous turns' reasoning back, since
the chat template renders it.

## Other checkpoints

`WEIGHTS=ORG/NAME[@REVISION]` serves another NVFP4 checkpoint of Qwen3.8-27B from the Hugging Face cache. The engine
takes the layouts it already reads: the modelopt one of NVIDIA's checkpoint (projections in FP8 or NVFP4) and the
compressed-tensors one of QUASAR, with NVFP4 activations (W4A4) and the model's dimensions. A BF16 `lm_head` (kept by
QUASAR and its fine-tunes) is quantized to NVFP4 at load, with the block scales chosen for the smallest error (on
QUASAR: KL to the BF16 model 0.0473, against 0.0471 with NVIDIA's NVFP4 `lm_head`). The compressed-tensors
"mixed-precision" layout of unsloth's and Red Hat's checkpoints loads as well: FP8 weights with per-output-channel
scales (the GDN and attention projections, the last eight MLPs, unsloth's `lm_head`) run as they are, with the
activations quantized per token at run time as the checkpoint specifies (`gemm_fp8_scaled`, `skinny_fp8_scaled`);
an FP8 `lm_head` gives the logits itself, and the DFlash2 draft reads an NVFP4 copy of it made at load (with the
draft vocabulary, only to build its own head: the copy then lives in the KV pool during the load and takes no VRAM
afterwards). Weight-only checkpoints
(NVFP4A16), GGUF files, other engines' formats and checkpoints that mix in other types are refused with the tensor at
fault. `scripts/download_weights.py` runs the engine's checks on the safetensors headers before it downloads anything
(it needs the engine library, `Q27_LIB`).

The stats page lists the downloaded checkpoints of the model under the start settings. The server keeps the tokenizer
of the weights it started with, so it starts an engine only on weights whose tokenizer has the same ids (vocabulary,
merges, added tokens); for a checkpoint with another one, start the server with `WEIGHTS` set to it. A
`tokenizer.json` that another transformers version rewrote (e.g. Huihui's abliterated checkpoint: another
pre-tokenizer pattern) counts as the same, and the server keeps its own. Prompts are rendered with the chat template
of the weights the engine runs (e.g. unsloth's, which merges several system messages and takes conversations without
a user message), switched with the engine. The DFlash2 draft was trained on the original
model: it stays correct on a fine-tune (the target model checks every token), but accepts fewer tokens the further
the fine-tune drifts.

## Windows package

The package is portable: what it writes stays in its folder (the one with `qwen3-27b-tray.exe`), so it needs a folder
you can write to, not Program Files. The exception is the start at logon, a value in the registry, only if you tick it.

| Path in the folder | What |
| --- | --- |
| `settings.ini` | Your settings (below); the package does not ship one, so an update does not touch it |
| `huggingface\` | The weights, 23 GB (`HF_HOME`) |
| `logs\` | The tray app's `server.log`, and one older file, `server.1.log`: a new one at each start of the server and beyond 16 MiB |
| `python\`, `app\`, `qwen3-27b-tray.exe`, `start.cmd`, `settings.example.ini`, `LICENSE` | The package; an update replaces them |

### Settings

Copy `settings.example.ini` to `settings.ini` and set any variable of this page in its `[env]` section ([How settings
work](#how-settings-work)). Both the tray app and `start.cmd` read it, when the server starts; the log says when
`settings.ini` overrides the environment. Relative paths are from the folder. The stats page does not write the file:
what you change there lasts until the server restarts, and the page shows the lines to add here to keep it. Only the
tray app's settings window writes it, when you click **Apply**, and only the lines of what you chose or typed there, also
a value equal to the default, which then stays when a later version changes the default (comments and other lines stay,
and it is saved as UTF-8).

`HF_HOME` is the exception: without it in `settings.ini` the weights are in the folder, whatever the environment says, so
that a variable of the whole PC does not move them out. To share an existing cache, write it there in full, for
example the one earlier versions downloaded to:

```ini
[env]
HF_HOME=C:\Users\me\.cache\huggingface
PORT=8001
```

Windows limits most paths to 260 characters and the weights' are about 140 below the folder: keep the folder's path under
about 110, or `HF_HOME` elsewhere.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPEN_UI` | 1 | `start.cmd`: 1 opens the web UI in the browser at the start, 0 does not |
| `WEIGHTS` | gdnfp4 | Weights downloaded before the start and served: a profile or `ORG/NAME[@REVISION]` ([Other checkpoints](#other-checkpoints)) |
| `HF_HOME` | `huggingface` in the folder | Hugging Face cache the weights are downloaded to and loaded from (23 GB) |
| `HF_TOKEN` | (none) | Sent with the downloads if set (needed for gated repositories; otherwise it only raises the rate limit) |

### Tray app

`qwen3-27b-tray.exe` does the same as `start.cmd` without a console: it runs `scripts/download_weights.py`, then the
server, in a job object, so that the server and the engine end with the app, even if the app is killed. A second start
of the app does nothing. If another server answers on the port, the app reports it and does not start one.

Its defaults differ from `start.cmd`: `ENGINE_AUTOSTART=0` and `ENGINE_IDLE_UNLOAD_S=1800`, so the service is up from the
start but holds VRAM only between the first request and 30 minutes after the last one ([Unloading when idle](#unloading-when-idle)).

The output goes to the log (the menu's **Show log**, which also works when the server did not start). **Start at
logon** is the value `qwen3-27b-5090` under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`; the app corrects its
path if the folder moved.

### Settings window

**Settings** in the tray menu edits `settings.ini`, without a browser:

| Group | Settings | Apply |
| --- | --- | --- |
| Server | `HOST`, `PORT`, `HF_HOME` | Restarts the server, which downloads the weights if the new `HF_HOME` does not have them |
| Engine | `WEIGHTS`, `KV_DTYPE`, `SPEC`, `VISION`, `MAX_SEQS`, `MAX_TOKENS`, `VRAM_BUDGET_GIB` | Reloads the engine if it is loaded (weights with another tokenizer: restarts the server); otherwise the next load uses them |
| When it holds the VRAM | `ENGINE_IDLE_UNLOAD_S` | Applies at once |
| When it holds the VRAM | `ENGINE_AUTOSTART`, `MIN_KV_TOKENS` | Restarts the server |

- The window says what **Apply** will do before you click it, and asks first if that fails requests in flight or empties
  the prefix cache. The values are saved even if the server then refuses them (for example, too little VRAM free now).
- Each engine choice shows the longest context and the KV pool it would leave, from the same planner as the stats page;
  the line below the choices sums up the settings as chosen. Without a server for that folder, `frontend/plan.py` gives
  the same answer (no GPU, nothing allocated).
- **Browse...** chooses the **Weights folder** in Explorer's dialog: the folder with `hub\` in it (choosing `hub\` itself
  counts as the folder above). **Weights** lists the checkpoints of the model there, also those of other repositories
  ([Other checkpoints](#other-checkpoints)), and follows the folder at once, before you apply it, with the estimates of
  that folder's weights. If the chosen weights or the DFlash2 draft are not there, the window says so (the draft's absence
  also means no estimate until `SPEC` is `none`) and asks first, as the server would download them.
- To share one cache with WSL, set `HF_HOME` to it on the Windows drive. `huggingface_hub` or `hf download` under WSL with
  a 9p mount of the drive leaves links (LX symlinks) that Windows programs cannot read; the window names those
  checkpoints, and **Fix the links** replaces the links of this model's checkpoints with hard links to the same files
  (`python\python.exe app\scripts\download_weights.py --fix-links`, with `HF_HOME` set). Both systems then read them as
  plain files, and nothing is copied or downloaded. It needs the links' files on an NTFS volume. A virtiofs mount
  (`virtiofs=true` in `.wslconfig`) and `scripts/download_weights.py` leave nothing to fix.
- All of them can be changed with the server stopped or failed, to repair a wrong port or folder.
- If the stats page changed something that `settings.ini` does not have, the window lists it, and **Use these** copies it
  into the window.
- Any other variable: **Open settings.ini**. The window reads the file when it opens.

### Updating

From v0.3.1, run `update.cmd` in the package's folder. It downloads the latest release's zip (an interrupted download
resumes when run again), checks its SHA-256, quits the tray app of this folder, replaces `python\`, `app\` and the
files next to them, and starts the tray app again if it was running. `settings.ini`, the weights and the log stay. If a
step fails, the old version is put back. A server started with `start.cmd` has to be closed by hand.

| Command | |
| --- | --- |
| `update.cmd` | The latest release (nothing to do when `app\VERSION` already has its version) |
| `update.cmd -Check` | Only show the installed version and the latest release's |
| `update.cmd -Tag v0.3.1` | That release, also an older one |
| `update.cmd -Force` | Install again when the version is the same |
| `update.cmd -Zip <file>` | A zip already downloaded (not checked against a SHA-256) |

By hand (and from v0.3.0, which has no `update.cmd`): quit the tray app, delete `python\` and `app\` (files of the old
version must not stay among the new ones), and unzip the new package over the folder (the zip's name has the version,
the folder in it does not).

## Development switches

All default to the faster setting. Except those marked static, they can be switched within one process with
`Engine.configure()` (`bench_spec.py --ab`).

| Variable | Meaning |
| --- | --- |
| `Q27_PROFILE=1` | GPU time per operation, also inside CUDA graphs (`Engine.profile_dump()`) |
| `Q27_NO_GRAPH=1` / `Q27_NO_PDL=1` / `Q27_NO_SKINNY=1` | Disable CUDA graphs / programmatic dependent launch / the skinny GEMMs |
| `Q27_SKINNY_CHECK=1` | Compare the skinny GEMMs with the prefill GEMMs and log the difference |
| `Q27_SKINNY_EXACT=1` | Make a row's skinny-GEMM result independent of M (tests of tree verification and preemption) |
| `Q27_SKINNY_EXT_REDUCE_M` | Rows from which the skinny GEMM's split-K reduction runs as a separate kernel (default 1: M >= 2; at M = 1 the last CTA reduces; same result; P7-4) |
| `Q27_DRAFT_ATTN_TC=0` | The draft's attention on CUDA cores (default: tensor cores; P7-4) |
| `Q27_SKIP=norm,silu,gdn,attn,gemm,gemm4,gemm8` | Skip a kind of kernel to measure its share of the time (`gemm4` / `gemm8`: FP4 / FP8 GEMMs only; the output is garbage) |
| `Q27_DECODE_MEMCPY=1` | Decode inputs and outputs by copies instead of mapped pinned memory |
| `Q27_ATTN_TC=0` | Decode / verify attention on CUDA cores (tree verification always uses them) |
| `Q27_ATTN_TC_DEC=0` | Decode (T = 1) attention with one warp per token (same result) |
| `Q27_ATTN_TC8_GRP=0` | With `Q27_ATTN_TC_F8=1`, compute T <= 2 in one warp instead of four (same result; static) |
| `Q27_KV_BLOCK=N` | Pages per block of the KV pool (P7-13; default the smallest count whose slice is >= 2 MiB: 57 for NVFP4, 32 for FP8, 16 for BF16). 1 lays pages out one by one (long-context decode 1.5% slower). Read at startup |
| `Q27_PAGES_HIGH=1` | Hand out KV pages from the top (tests, P7-13): the KV cache then occupies the parking areas, so every park moves pages. Read at startup |
| `Q27_ATTN_PREFILL_TC=0` | Prefill attention on CUDA cores (static) |
| `Q27_GDN_CHUNK=0` | Sequential GDN in prefill (static) |
| `Q27_MLP_FUSED=0` | Prefill MLP: gate \| up as a BF16-output GEMM, then SiLU and NVFP4 quantization in a separate kernel. Default 1: one GEMM (`gemm_mma.cu`) whose epilogue writes silu(gate) * up as the down GEMM's NVFP4 activations (prefill +2% at 19K / 73K, P5-12) |
| `Q27_MLP_BF16=0` | With `Q27_MLP_FUSED=0`: fp32 outputs for the prefill MLP's gate \| up |
| `Q27_SPEC_PATHS=K` | Paths of tree verification (default 1 = a chain; 2 / 4 accept more but take longer and break even) |
| `Q27_LOOKUP=1` | Lookup drafting (off: no gain with the 8-row chain) |
| `Q27_SPEC_PIPELINE=0` | Enqueue a speculative step after reading the previous result (default 1: before) |
| `Q27_SKINNY_WAVES` / `Q27_SKINNY_FIT` / `Q27_SKINNY_L2_CHUNKS` / `Q27_ATTN_WAVES` / `Q27_ATTN_MIN_KEYS` / `Q27_ATTN_PREFETCH` / `Q27_ATTN_TC_SPLITS` | Tuning |
| `Q27_LIB` / `Q27_TRACE` | Another build of `libq27.so` (e.g. for tracing; the default is `/build/engine/libq27.so`, on Windows `engine/build/q27.dll`) / where a `-DQ27_TRACE` build writes its timeline |
