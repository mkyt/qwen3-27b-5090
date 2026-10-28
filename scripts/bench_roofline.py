"""Measure runtime/kernel throughput and NVML telemetry without changing the power limit.

Run inside the dev container; use --mode kernels or runtime. Raw repetitions and
telemetry are retained, rather than selecting the fastest run. Runtime results
exclude model loading, reset and prefix-cache reuse. No speculative decoding.
"""

import argparse
import ctypes as ct
import json
import os
import pathlib
import statistics
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))


class Telemetry:
    def __init__(self):
        self.nv = ct.CDLL("libnvidia-ml.so.1")
        self.check(self.nv.nvmlInit_v2())
        self.handle = ct.c_void_p()
        self.check(self.nv.nvmlDeviceGetHandleByIndex_v2(0, ct.byref(self.handle)))
        self.done = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"NVML returned {status}")

    def value(self, function, dtype=ct.c_uint, *args):
        value = dtype()
        self.check(getattr(self.nv, function)(self.handle, *args, ct.byref(value)))
        return value.value

    def sample(self):
        while not self.done.is_set():
            try:
                self.samples.append({
                    "time": time.perf_counter(),
                    "power_w": self.value("nvmlDeviceGetPowerUsage") / 1000,
                    "sm_mhz": self.value("nvmlDeviceGetClockInfo", ct.c_uint, 1),
                    "mem_mhz": self.value("nvmlDeviceGetClockInfo", ct.c_uint, 2),
                    "reasons": self.value("nvmlDeviceGetCurrentClocksEventReasons", ct.c_ulonglong),
                })
            except Exception as error:
                self.samples.append({"time": time.perf_counter(), "error": str(error)})
            self.done.wait(0.02)

    def energy(self):
        return self.value("nvmlDeviceGetTotalEnergyConsumption", ct.c_ulonglong) / 1000

    def measure(self, function):
        e0 = self.energy()
        start = time.perf_counter()
        result = function()
        end = time.perf_counter()
        joules = self.energy() - e0
        samples = [s for s in self.samples if start <= s["time"] <= end]
        good = [s for s in samples if "error" not in s]
        row = {"wall_s": end - start, "energy_j": joules,
               "energy_mean_w": joules / (end - start), "samples": samples}
        if good:
            row.update({"sm_mhz": statistics.mean(s["sm_mhz"] for s in good),
                        "mem_mhz": statistics.mean(s["mem_mhz"] for s in good),
                        "power_cap_fraction": statistics.mean(bool(s["reasons"] & 4) for s in good)})
        return result, row

    def close(self):
        self.done.set()
        self.thread.join()
        self.check(self.nv.nvmlShutdown())


def emit(row):
    print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)


def runtime(args, telemetry):
    from q27lib import Engine
    from qwen27b.weights import engine_target
    lengths = sorted(set(args.prefill + args.decode))
    eng = Engine(engine_target("nvidia"), max_tokens=max(lengths) + args.tokens + 64,
                 prefill_chunk=2048, kv_dtype=args.kv_dtype)
    base = max((json.loads(line) for line in open(ROOT / "bench/golden/prompts.jsonl")),
               key=lambda p: len(p["ids"]))["ids"]
    rows = []
    try:
        eng.forward_sample(base[:256])
        for n in lengths:
            ids = (base * (n // len(base) + 1))[:n]
            for rep in range(args.reps):
                eng.reset()
                time.sleep(0.3)
                nxt, row = telemetry.measure(lambda: eng.forward_sample(ids))
                row.update(kind="prefill", ctx=n, rep=rep, next_token=nxt,
                           tok_s=n / row["wall_s"], joules_per_token=row["energy_j"] / n)
                rows.append(row)
                emit(row)
                if n in args.decode:
                    # Capture/warm the decode graph before measuring steady-state steps.
                    for _ in range(8):
                        nxt = eng.forward_sample([nxt])
                    steps = []

                    def decode():
                        nonlocal nxt
                        for _ in range(args.tokens):
                            start = time.perf_counter()
                            nxt = eng.forward_sample([nxt])
                            steps.append(time.perf_counter() - start)

                    _, row = telemetry.measure(decode)
                    row.update(kind="decode", ctx=n, rep=rep, tokens=args.tokens,
                               context_midpoint=n + 8 + (args.tokens - 1) / 2,
                               tok_s=args.tokens / row["wall_s"],
                               median_ms=statistics.median(steps) * 1000,
                               p90_ms=sorted(steps)[int(0.9 * len(steps))] * 1000,
                               joules_per_token=row["energy_j"] / args.tokens)
                    rows.append(row)
                    emit(row)
    finally:
        eng.close()
    return rows


def kernels(args, telemetry):
    import torch
    from q27lib import call, lib
    torch.manual_seed(7)
    rows = []
    for precision, name, m, n, k, bf16_out in [
        ("fp4", "gate_up_bf16", 2048, 17408, 5120, True),
        ("fp4", "down_f32", 2048, 5120, 17408, False),
        ("fp8", "gdn_qkv", 2048, 10240, 5120, False),
        ("fp8", "gdn_out", 2048, 5120, 6144, False),
        ("fp8", "attn_kv", 2048, 1024, 5120, False),
    ]:
        if precision == "fp4":
            a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device="cuda")
            b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
            sf = []
            for r in (m, n):
                src = torch.full((r, k // 16), 0x38, dtype=torch.uint8, device="cuda")
                dst = torch.empty(lib().q27t_sf_bytes(r, k // 16), dtype=torch.uint8, device="cuda")
                call("q27t_swizzle_sf", src, r, k // 16, dst)
                sf.append(dst)
            out = torch.empty(m, n, dtype=torch.bfloat16 if bf16_out else torch.float32, device="cuda")
            if bf16_out:
                def run():
                    call("q27t_gemm_fp4_bf16", m, n, k, a, sf[0], b, sf[1], 1.0, out)
            else:
                def run():
                    call("q27t_gemm_fp4", m, n, k, a, sf[0], b, sf[1], 1.0, None, 0.0, out)
        else:
            a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
            b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
            out = torch.empty(m, n, device="cuda")

            def run():
                call("q27t_gemm_fp8", m, n, k, a, b, 1.0, None, 0.0, out)

        for _ in range(100):
            run()
        torch.cuda.synchronize()
        # q27t wrappers use a private stream and synchronize each call. Their
        # built-in graph timer measures that stream and removes Python launch gaps.
        os.environ["Q27T_REPEAT"] = "200"
        run()
        iterations = max(100, int(1500 / (lib().q27t_last_us() / 1000)))
        os.environ["Q27T_REPEAT"] = str(iterations)
        for rep in range(args.reps):
            def measure():
                run()
                return lib().q27t_last_us() / 1000

            ms, row = telemetry.measure(measure)
            row.update(kind="gemm", precision=precision, name=name, M=m, N=n, K=k,
                       rep=rep, iterations=iterations, gpu_ms=ms,
                       measurement="GPU graph replay; telemetry includes capture and warmup",
                       tflops=2 * m * n * k / (ms * 1e9))
            rows.append(row)
            emit(row)
        del os.environ["Q27T_REPEAT"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["runtime", "kernels"], required=True)
    ap.add_argument("--prefill", type=lambda s: list(map(int, s.split(","))), default=[19000, 32768, 73000, 194000])
    ap.add_argument("--decode", type=lambda s: list(map(int, s.split(","))), default=[128, 4096, 32768, 131072])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--kv-dtype", choices=["fp8", "nvfp4"], default="fp8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    telemetry = Telemetry()
    try:
        limit = telemetry.value("nvmlDeviceGetPowerManagementLimit") / 1000
        if limit != 400:
            raise RuntimeError(f"Expected unchanged 400 W power limit, found {limit}")
        rows = (runtime if args.mode == "runtime" else kernels)(args, telemetry)
        args.out.write_text(json.dumps({"mode": args.mode, "reps": args.reps,
                                       "kv_dtype": args.kv_dtype,
                                       "power_limit_w": limit, "rows": rows}, indent=2) + "\n")
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
