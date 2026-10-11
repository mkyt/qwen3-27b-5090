"""Phase Q: quantize only the GDN / attention projections of the BF16 original to NVFP4 (allfp4, docs/decisions.md "Weights").

The MLP, lm_head and every other tensor stay those of the nvidia checkpoint. The output directory holds one new
shard with the projections in the modelopt layout (weight, weight_scale, weight_scale_2, input_scale) and
symlinks to the nvidia shards; the new shard's name sorts first, so the engine (which takes the first shard
that has a name) and the index (which maps the projection names to it) both read the new projections.

    python scripts/quantize_proj.py hessian --out cache/pq-hessian [--docs FILE...] [--dumps DIR] [--max-tokens 400000]
    python scripts/quantize_proj.py build --method rtn|mse|gptq [--hessians cache/pq-hessian] --out cache/weights/allfp4-<name>

Activation scales: input_scale = (the nvidia FP8 input_scale) / 6, i.e. the same calibrated amax (448 x the
FP8 scale) mapped to the NVFP4 range (6 x 448).
Methods: quasar (the QAT projections of QUASAR, only converted to the modelopt layout), rtn (block scale amax / 6), mse (per block, the e4m3 scale near amax / 6 with the smallest error,
weighted by diag(H) when Hessians are given), gptq (error compensation with H = sum x xT of the projection
inputs; the block scales are chosen as in mse when the column loop reaches each 16-column block).
Calibration set (hessian): the --docs files, this repository's code and the --dumps conversations, in sequences of
--seq-len plus one of --long-len. Arguments can be read from a file, one per line: `hessian @calib.args`.
"""

import argparse
import json
import re
import os
import pathlib
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reference"))
from qwen27b.quant import E2M1_MAX, E4M3_MAX, FP4_BLOCK, dequant_nvfp4, round_e2m1, to_e4m3  # noqa: E402
from qwen27b.weights import BF16, QUASAR, TARGET, Checkpoint, snapshot  # noqa: E402

LM = "model.language_model"
SHARD = "model-00000-proj-nvfp4.safetensors"
GDN = ("in_proj_qkv", "in_proj_z", "out_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def projections(cfg):
    """(layer, prefix, hessian key) of every projection; the key names the input the projection reads."""
    for i, kind in enumerate(cfg["text_config"]["layer_types"]):
        if kind == "linear_attention":
            for n in GDN:
                yield i, f"{LM}.layers.{i}.linear_attn.{n}", f"{i:02d}-{'out' if n == 'out_proj' else 'in'}"
        else:
            for n in ATTN:
                yield i, f"{LM}.layers.{i}.self_attn.{n}", f"{i:02d}-{'out' if n == 'o_proj' else 'in'}"


FUSED = (("in_proj_qkv", "in_proj_z"), ("q_proj", "k_proj", "v_proj"))  # projections other engines run as one GEMM


def fused_group(prefix):
    """The other projections fused with `prefix` (same input) in vLLM / SGLang."""
    head, name = prefix.rsplit(".", 1)
    for g in FUSED:
        if name in g:
            return [f"{head}.{n}" for n in g if n != name]
    return []


# --- quantization -------------------------------------------------------------------------------------------

def e4m3_from_code(c: torch.Tensor) -> torch.Tensor:
    return c.to(torch.uint8).view(torch.float8_e4m3fn).float()


def block_error(wb, sf, gscale, hd):
    """wb [..., 16] weights, sf [...] candidate block scales -> weighted squared error [...]."""
    s = (sf * gscale).unsqueeze(-1)
    q = round_e2m1(wb / torch.where(s > 0, s, torch.ones_like(s))) * s
    return ((wb - q) ** 2 * hd).sum(-1)


def choose_scales(wb, gscale, hd, search):
    """wb [rows, nb, 16] -> e4m3 block scales [rows, nb] (fp32 values). search: e4m3 codes tried below / above
    the rounded amax / 6 (0: plain rtn)."""
    sf0 = to_e4m3(wb.abs().amax(-1) / E2M1_MAX / gscale)
    if search == 0:
        return sf0
    c0 = sf0.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    best, best_err = sf0, block_error(wb, sf0, gscale, hd)
    for d in range(-search, 2):
        if d == 0:
            continue
        sf = e4m3_from_code((c0 + d).clamp(1, 126))
        sf = torch.where(sf0 > 0, sf, sf0)
        err = block_error(wb, sf, gscale, hd)
        better = err < best_err
        best, best_err = torch.where(better, sf, best), torch.where(better, err, best_err)
    return best


def pack(w, sf, gscale):
    """w [out, in] fp32 with block scales sf [out, in/16] -> (uint8 [out, in/2], e4m3 [out, in/16])."""
    s = (sf * gscale).repeat_interleave(FP4_BLOCK, dim=1)
    q = round_e2m1(w / torch.where(s > 0, s, torch.ones_like(s)))
    mag = torch.bucketize(q.abs(), torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device))
    codes = (mag | ((q < 0).to(torch.long) << 3)).to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), sf.to(torch.float8_e4m3fn)


def quant_rtn(w, gscale, hdiag, search, rows=1024):
    out, nin = w.shape
    sfs = []
    hd = (hdiag if hdiag is not None else torch.ones(nin, device=w.device)).view(nin // FP4_BLOCK, FP4_BLOCK)
    for r in range(0, out, rows):
        sfs.append(choose_scales(w[r:r + rows].view(-1, nin // FP4_BLOCK, FP4_BLOCK), gscale, hd, search))
    return torch.cat(sfs)


def quant_gptq(w, gscale, H, search, damp=0.01, bs=128):
    w = w.clone()
    out, nin = w.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    w[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(nin, device=w.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    hdiag = torch.diag(H)
    sf = torch.empty(out, nin // FP4_BLOCK, device=w.device)
    Q = torch.empty_like(w)
    for i1 in range(0, nin, bs):
        i2 = min(i1 + bs, nin)
        W1, Hi = w[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if i % FP4_BLOCK == 0:
                b = (i1 + i) // FP4_BLOCK
                sf[:, b] = choose_scales(W1[:, i:i + FP4_BLOCK].unsqueeze(1), gscale,
                                         hdiag[i1 + i:i1 + i + FP4_BLOCK].view(1, FP4_BLOCK), search)[:, 0]
                s = sf[:, b] * gscale
                s1 = torch.where(s > 0, s, torch.ones_like(s))
            col = W1[:, i]
            q = round_e2m1(col / s1) * s
            Q[:, i1 + i] = q
            e = (col - q) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = e
        w[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return sf, Q


def cmd_build(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nv_dir = snapshot(*TARGET)
    nv = Checkpoint(nv_dir)
    bf = Checkpoint(snapshot(*(QUASAR if args.method == "quasar" else BF16)))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tensors, stats = {}, []
    t0 = time.time()
    keep = re.compile(args.keep_fp8) if args.keep_fp8 else None
    for i, p, key in projections(nv.config):
        if args.layers and i not in args.layers:
            continue
        if keep and keep.search(p):  # stays the nvidia FP8 projection
            continue
        if args.method == "quasar":  # QUASAR's QAT projections, compressed-tensors -> modelopt (no rounding)
            tensors[f"{p}.weight"] = bf.get(f"{p}.weight_packed")
            tensors[f"{p}.weight_scale"] = bf.get(f"{p}.weight_scale")
            tensors[f"{p}.weight_scale_2"] = (1 / bf.get(f"{p}.weight_global_scale").float()).reshape(())
            tensors[f"{p}.input_scale"] = (1 / bf.get(f"{p}.input_global_scale").float()).reshape(())
            continue
        part = pathlib.Path(args.parts) / f"{p}.pt" if args.parts else None
        if part and part.is_file():  # quantized by an earlier run (layer groups as their Hessians arrive)
            d = torch.load(part)
            tensors.update(d["tensors"])
            stats.append(d["stats"])
            print(f"{p}: from {part}", flush=True)
            continue
        w = bf.get(f"{p}.weight", dev).float()
        amax = w.abs().amax()
        if args.shared_global:  # one weight global scale per fused group (vLLM keeps only the largest, P-Q export)
            for q in fused_group(p):
                amax = torch.maximum(amax, bf.get(f"{q}.weight", dev).float().abs().amax())
        gscale = amax / (E2M1_MAX * E4M3_MAX)
        H = None
        if args.hessians:
            d = torch.load(pathlib.Path(args.hessians) / f"{key}.pt", map_location=dev)
            H = d["H"] / d["n"]
        if args.method == "gptq":
            sf, wq = quant_gptq(w, gscale, H, args.search, args.damp)  # wq: on the grid, pack() keeps it
        else:
            sf, wq = quant_rtn(w, gscale, torch.diag(H) if H is not None else None,
                               args.search if args.method == "mse" else 0), w
        packed, sf8 = pack(wq, sf, gscale)
        dq = dequant_nvfp4(packed, sf8, gscale)
        rel = ((dq - w).norm() / w.norm()).item()
        out_err = ((((dq - w) @ H) * (dq - w)).sum() / ((w @ H) * w).sum()).item() if H is not None else float("nan")
        stats.append({"name": p, "rel": rel, "out_rel": out_err})
        tensors[f"{p}.weight"] = packed.cpu()
        tensors[f"{p}.weight_scale"] = sf8.cpu()
        tensors[f"{p}.weight_scale_2"] = gscale.reshape(()).float().cpu()
        tensors[f"{p}.input_scale"] = (nv.get(f"{p}.input_scale").float() / E2M1_MAX).reshape(())
        print(f"{p}: rel {rel:.4f} out {out_err:.5f} ({time.time() - t0:.0f} s)", flush=True)
        if part:
            part.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"tensors": {k: v for k, v in tensors.items() if k.startswith(p + ".")}, "stats": stats[-1]}, part)
    if not stats:
        stats = [{"name": "(quasar projections)", "rel": float("nan"), "out_rel": float("nan")}]
    meta = {"method": args.method, "search": str(args.search), "keep_fp8": args.keep_fp8,
            "shared_global": str(args.shared_global), "hessians": str(args.hessians or ""),
            "source": "{}@{}".format(*(QUASAR if args.method == "quasar" else BF16)), "base": f"{TARGET[0]}@{TARGET[1]}"}
    save_file(tensors, out / SHARD, metadata=meta)
    # The rest: the nvidia checkpoint (symlinks, relative when both are under cache/weights).
    wm = json.loads((nv_dir / "model.safetensors.index.json").read_text())
    drop = {n for _, p, _ in projections(nv.config) for n in (f"{p}.weight", f"{p}.weight_scale", f"{p}.input_scale")
            if f"{p}.weight" in tensors}
    wm["weight_map"] = {k: v for k, v in wm["weight_map"].items() if k not in drop}
    wm["weight_map"].update({k: SHARD for k in tensors})
    wm["metadata"] = {"allfp4": meta}
    for f in nv_dir.iterdir():
        if f.name == "model.safetensors.index.json":
            continue
        link = out / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(f.resolve(), out.resolve()))
    (out / "model.safetensors.index.json").write_text(json.dumps(wm, indent=2))
    (out / "quant_stats.json").write_text(json.dumps({"meta": meta, "layers": stats}, indent=1))
    print(f"wrote {out} ({len(tensors) // 4} projections, mean rel {sum(s['rel'] for s in stats) / len(stats):.4f})")


# --- calibration --------------------------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parents[1]
# Not the golden sources (bench/golden/sources) nor any evaluation data. Documents and API dumps are the user's own
# (--docs, --dumps): text like the requests the server will see.
CALIB_CODE = ["frontend/*.py", "engine/src/*.cc", "engine/src/*.cu", "engine/src/*.h", "scripts/needle_test.py",
              "scripts/soak_test.py", "scripts/quality_tools.py", "reference/qwen27b/*.py"]


def calib_sequences(tok, seq_len, long_len, docs, dumps, max_code=14):
    """Token id lists: documents, code, agent requests with tools (API dumps), one long one."""
    sys.path.insert(0, str(HERE / "frontend"))
    import chat

    def chunks(text, n):
        ids = tok(text, add_special_tokens=False).input_ids
        return [ids[i:i + n] for i in range(0, len(ids), n) if len(ids) - i >= 256]

    docs = "\n\n".join(pathlib.Path(d).read_text() for d in docs)
    code = "\n\n".join(f"# {f.relative_to(HERE)}\n{f.read_text()}" for g in CALIB_CODE for f in sorted(HERE.glob(g)))
    seqs = [("long", chunks(docs + "\n\n" + code, long_len)[0])]
    seqs += [("doc", c) for c in chunks(docs, seq_len)] + [("code", c) for c in chunks(code, seq_len)][:max_code]
    threads = {}  # an agent session sends its growing conversation again and again: keep the longest
    for f in sorted(pathlib.Path(dumps).glob("*.json")) if dumps else []:
        body = json.loads(f.read_text())
        try:
            if f.name.endswith("_v1_messages.json"):
                msgs, tools = chat.anthropic_messages(body)
            elif f.name.endswith("_v1_chat_completions.json"):
                msgs, tools = chat.normalize_messages(body["messages"]), body.get("tools")
            else:
                continue
            text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=False)
        except Exception as e:  # noqa: BLE001 - a dump the template does not take
            print(f"skip {f.name}: {e}", file=sys.stderr)
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        key = json.dumps(msgs[:2], sort_keys=True, default=str)
        if len(ids) >= 256 and len(ids) > len(threads.get(key, [])):
            threads[key] = ids
    for ids in threads.values():  # the start (system prompt, tools) and the latest turns
        seqs.append(("agent", ids[:seq_len]))
        if len(ids) > seq_len:
            seqs.append(("agent", ids[-seq_len:]))
    return seqs


def cmd_hessian(args):
    from transformers import AutoTokenizer

    from qwen27b.model import Qwen38Reference, SeqState

    nv = Checkpoint(snapshot(*TARGET))
    tok = AutoTokenizer.from_pretrained(snapshot(*TARGET))
    seqs = calib_sequences(tok, args.seq_len, args.long_len, args.docs, args.dumps)
    total = 0
    for i, (kind, ids) in enumerate(seqs):
        if total >= args.max_tokens:
            seqs = seqs[:i]
            break
        total += len(ids)
    print({k: sum(len(s) for kk, s in seqs if kk == k) for k in ("long", "doc", "code", "agent")}, f"total {total}", flush=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = {}  # the projection that reads each input first (q / k / v and qkv / z read the same input)
    for i, p, key in projections(nv.config):
        if (not args.layers or i in args.layers) and key not in first.values():
            first[p] = key
    ref = Qwen38Reference(nv, device="cuda", kv_dtype="fp8")
    H, n = {}, {}
    orig = ref._linear

    def linear(prefix):
        lin, key = orig(prefix), first.get(prefix)
        if key is None:
            return lin

        def call(x):
            xf = x.reshape(-1, x.shape[-1]).float()
            if key not in H:
                H[key] = torch.zeros(xf.shape[1], xf.shape[1], device=xf.device)
                n[key] = 0
            H[key].addmm_(xf.T, xf)
            n[key] += xf.shape[0]
            return lin(x)
        return call

    ref._linear = linear
    t0 = time.time()
    with torch.no_grad():
        for j, (kind, ids) in enumerate(seqs):
            ref.forward(torch.tensor(ids), SeqState(), return_hidden=True)  # (no lm_head: 15 GB of temporaries)
            print(f"{j + 1}/{len(seqs)} {kind} {len(ids)} ({time.time() - t0:.0f} s)", flush=True)
    for key, h in H.items():
        torch.save({"H": h.cpu(), "n": n[key]}, out / f"{key}.pt")
    (out / "calib.json").write_text(json.dumps({"tokens": total, "sequences": [[k, len(s)] for k, s in seqs]}))
    print(f"wrote {len(H)} Hessians to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--method", choices=["rtn", "mse", "gptq", "quasar"], default="rtn",
                   help="quasar: QUASAR-QAT's projections as they are (with the nvidia MLP)")
    b.add_argument("--search", type=int, default=3, help="mse / gptq: e4m3 codes tried below amax / 6")
    b.add_argument("--hessians", default="", help="directory of the `hessian` command (mse: diag weights; gptq: required)")
    b.add_argument("--damp", type=float, default=0.01)
    b.add_argument("--shared-global", action="store_true",
                   help="one weight global scale per fused group (qkv | z, q | k | v): needed by vLLM, which keeps the max")
    b.add_argument("--keep-fp8", default="", help="regex: projections (prefix names) left as the nvidia FP8 ones")
    b.add_argument("--parts", default="", help="keep / reuse each projection's result here (run layer groups early)")
    b.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[], help="(testing) only these")
    b.add_argument("--out", required=True)
    h = sub.add_parser("hessian")
    h.add_argument("--out", required=True)
    h.add_argument("--docs", nargs="*", default=[], help="text files of the calibration set, in this order")
    h.add_argument("--dumps", default="", help="directory of API request bodies (<name>_v1_messages.json, "
                                              "<name>_v1_chat_completions.json): agent conversations with tools")
    h.add_argument("--max-tokens", type=int, default=400000)
    h.add_argument("--seq-len", type=int, default=8192)
    h.add_argument("--long-len", type=int, default=24576)
    h.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[], help="only these (memory)")
    args = ap.parse_args()
    if args.cmd == "hessian":
        cmd_hessian(args)
    if args.cmd == "build":
        if args.method == "gptq" and not args.hessians:
            ap.error("gptq needs --hessians")
        cmd_build(args)


if __name__ == "__main__":
    main()
