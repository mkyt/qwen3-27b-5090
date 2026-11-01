"""Download the checkpoints the server starts with into the Hugging Face cache, with the Python standard library only:
the Windows package runs it before the server (no huggingface_hub, uv or symlinks), and it works on Linux as well.

    python scripts/download_weights.py [--weights gdnfp4|nvidia|quasar|ORG/NAME[@REVISION]] [--no-draft] [--jobs 4]
                                       [--also nvidia|quasar|bf16|ORG/NAME[@REVISION]]...

--weights defaults to $WEIGHTS, else gdnfp4; the DFlash2 draft comes too unless --no-draft or SPEC=none. --also adds
more checkpoints (bf16: the BF16 original; scripts/download-weights.sh uses it for --dev, --bf16, --quasar). A profile's
revision is pinned (reference/qwen27b/paths.py); a repository's branch or tag (default main) is resolved to its
commit, recorded under refs/ as huggingface_hub does. Before downloading a repository, its safetensors headers (a few
hundred KB) go through the engine's own checks (q27_plan, no GPU), so an unsupported checkpoint stops here rather than
after 20 GB; the check needs the engine library (Q27_LIB), and is skipped without it.

huggingface_hub under WSL with a 9p mount of a Windows drive leaves links (LX symlinks) that Windows programs cannot
read. --fix-links, on Windows, replaces the links of the model's repositories with hard links to the same
blobs, which both systems read as plain files: nothing is copied or downloaded. It prints {"fixed": n, "failed": [paths]}.

The files go to $HF_HOME/hub/models--<org>--<name>/snapshots/<commit>/ (HF_HOME defaults to ~/.cache/huggingface),
where paths.py looks for them. Files already there with the right size are kept; a download resumes from its
.incomplete file (also on network errors, retried), and every LFS file is checked against its SHA-256 before it takes
its final name. HF_TOKEN, if set, is sent (needed for gated repositories).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from qwen27b import paths  # noqa: E402

HF = "https://huggingface.co"
CHUNK = 8 << 20
RETRIES = 8  # per file, on network errors (each resumes from the .incomplete file), 5 s doubling up to 2 min apart
PROFILE_PINS = {"gdnfp4": [paths.GDNFP4], "nvidia": [paths.TARGET], "quasar": [paths.QUASAR, paths.TARGET]}
# What the server reads besides the weights (frontend/tokenizer.py, launch.py); a chat template in
# chat_template.jinja or tokenizer_config.json.
SERVER_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def request(url, headers=None):
    h = {"User-Agent": "qwen3-27b-5090-download"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return urllib.request.Request(url, headers={**h, **(headers or {})})


def get(url, headers=None):
    with urllib.request.urlopen(request(url, headers), timeout=60) as r:
        return r.read()


def commit_of(repo, rev):
    """The commit of a branch, tag or commit."""
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    return json.loads(get(f"{HF}/api/models/{repo}/revision/{urllib.parse.quote(rev, safe='')}"))["sha"]


def files_of(repo, rev):
    """The files of a revision: [(path, size, sha256 or None)]."""
    tree = json.loads(get(f"{HF}/api/models/{repo}/tree/{rev}?recursive=true"))
    return [(t["path"], t["size"], t.get("lfs", {}).get("oid")) for t in tree if t["type"] == "file"]


def check_remote(repo, rev, files):
    """None if the engine accepts the checkpoint (its plan from the safetensors headers and config.json), else why
    not; "skipped: ..." without the engine library."""
    names = {p for p, _, _ in files}
    missing = [f for f in SERVER_FILES if f not in names]
    if missing:
        return f"missing {', '.join(missing)}"
    base = f"{HF}/{repo}/resolve/{rev}/"
    if "chat_template.jinja" not in names and "chat_template" not in json.loads(get(base + "tokenizer_config.json")):
        return "no chat template (chat_template.jinja or tokenizer_config.json)"
    sys.path.insert(0, str(ROOT / "engine/python"))
    try:
        import q27lib
        q27lib.lib()
    except (ImportError, OSError) as e:
        return f"skipped: no engine library ({e})"
    shards = sorted(p for p in names if p.endswith(".safetensors"))
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        for f in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            if f in names:
                (d / f).write_bytes(get(base + f))
        for f in shards:  # (header only: the plan reads no tensor data)
            url = base + urllib.parse.quote(f)
            n = struct.unpack("<Q", get(url, {"Range": "bytes=0-7"}))[0]
            (d / f).write_bytes(struct.pack("<Q", n) + get(url, {"Range": f"bytes=8-{7 + n}"}))
        saved = {k: os.environ.get(k) for k in ("VISION", "Q27_LM_HEAD_CKPT")}
        os.environ["VISION"] = "host"
        os.environ.pop("Q27_LM_HEAD_CKPT", None)
        try:
            q27lib.Engine.plan(str(d), 30 << 30, 32 << 30, max_tokens=32768, kv_dtype="nvfp4").close()
        except RuntimeError as e:
            return str(e).removeprefix("q27_plan: ")
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return None


class Progress:
    def __init__(self, total):
        self.total, self.done, self.lock, self.t0, self.last = total, 0, threading.Lock(), time.time(), 0.0

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= 5:
                self.last = now
                rate = self.done / max(now - self.t0, 1e-6) / 1e6
                print(f"  {self.done / 1e9:.1f} / {self.total / 1e9:.1f} GB ({rate:.0f} MB/s)", flush=True)


def fetch(repo, rev, path, size, sha, dst, progress):
    part = dst.with_name(dst.name + ".incomplete")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    h = hashlib.sha256()
    if sha and have:
        with part.open("rb") as f:
            while b := f.read(CHUNK):
                h.update(b)
    if have < size:
        url = f"{HF}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
        with urllib.request.urlopen(request(url, {"Range": f"bytes={have}-"} if have else None), timeout=60) as r:
            if have and r.status != 206:  # (no range support: start over)
                have, h = 0, hashlib.sha256()
            with part.open("ab" if have else "wb") as f:
                while b := r.read(CHUNK):
                    f.write(b)
                    h.update(b)
                    progress.add(len(b))
    if part.stat().st_size != size:
        raise OSError(f"{path}: got {part.stat().st_size} bytes, expected {size}")
    if sha and h.hexdigest() != sha:
        part.unlink()
        raise OSError(f"{path}: SHA-256 mismatch (removed; run again)")
    os.replace(part, dst)


def fetch_retrying(repo, rev, path, size, sha, dst, progress):
    for attempt in range(RETRIES + 1):
        try:
            return fetch(repo, rev, path, size, sha, dst, progress)
        except OSError as e:  # (urllib's errors, timeouts, a short read, a SHA-256 mismatch)
            if attempt == RETRIES:
                raise
            wait = min(120, 5 << attempt)
            print(f"  {path}: {e}; retrying in {wait} s", flush=True)
            time.sleep(wait)


IO_REPARSE_TAG_LX_SYMLINK = 0xA000001D


def lx_target(path):
    """The target of an LX symlink, which Windows keeps in the reparse point and does not follow (Windows only)."""
    import ctypes
    import ctypes.wintypes as w

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = w.HANDLE
    kernel32.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    kernel32.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, w.LPVOID, w.DWORD, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID]
    kernel32.CloseHandle.argtypes = [w.HANDLE]
    # (OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS; FSCTL_GET_REPARSE_POINT)
    handle = kernel32.CreateFileW(str(path), 0, 7, None, 3, 0x02200000, None)
    if handle in (None, w.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(16384)
    size = w.DWORD()
    done = kernel32.DeviceIoControl(handle, 0x900A8, None, 0, buffer, len(buffer), ctypes.byref(size), None)
    kernel32.CloseHandle(handle)
    if not done:
        raise ctypes.WinError(ctypes.get_last_error())
    tag, length = struct.unpack_from("<IH", buffer.raw)
    if tag != IO_REPARSE_TAG_LX_SYMLINK:
        raise OSError(f"{path}: not an LX symlink")
    return buffer.raw[12:8 + length].decode("utf-8")  # (header 8 bytes, version 4 bytes, the path in UTF-8)


def link_to_blob(path, target, base):
    """Replaces the link `path` with a hard link to its target, a file in the repository's directory `base`."""
    blob = pathlib.Path(os.path.normpath(path.parent / target))
    if base not in blob.parents or not blob.is_file():
        raise OSError(f"{path}: the target {target} is not a file of the repository")
    temporary = path.with_name(path.name + ".link-tmp")
    os.link(blob, temporary)
    os.replace(temporary, path)


def fix_links():
    """The LX symlinks of the model's repositories, as hard links to their blobs."""
    if os.name != "nt":
        sys.exit("--fix-links is for Windows, which tells the links it cannot read from those it can")
    fixed, failed = 0, []
    for base in sorted(paths.HF_HUB.glob("models--*--*")):
        if "qwen3.8-27b" not in base.name.lower():
            continue
        for path in base.glob("snapshots/*/**/*"):
            if getattr(os.lstat(path), "st_reparse_tag", 0) != IO_REPARSE_TAG_LX_SYMLINK:
                continue
            try:
                link_to_blob(path, lx_target(path), base)
                fixed += 1
            except OSError as e:
                failed.append(f"{path}: {e}")
    return {"fixed": fixed, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS") or "gdnfp4",
                    help="a profile (gdnfp4, nvidia, quasar) or a Hugging Face repository ORG/NAME[@REVISION]")
    ap.add_argument("--no-draft", action="store_true", default=os.environ.get("SPEC") == "none",
                    help="without the DFlash2 draft (speculative decoding off: SPEC=none)")
    ap.add_argument("--no-check", action="store_true", help="download a repository without the engine's check")
    ap.add_argument("--jobs", type=int, default=4, help="files downloaded at once")
    ap.add_argument("--also", action="append", default=[], metavar="WEIGHTS",
                    help="another checkpoint to download: nvidia, quasar, bf16 or ORG/NAME[@REVISION] (repeatable)")
    ap.add_argument("--fix-links", action="store_true", help="on Windows: the model's LX symlinks as hard links to their blobs")
    args = ap.parse_args()

    if args.fix_links:
        print(json.dumps(fix_links()))
        return

    pins = []
    for opt, w in [("--weights", args.weights)] + [("--also", a) for a in args.also]:
        if w in PROFILE_PINS or w == "bf16":
            new = [(repo, rev, None) for repo, rev in PROFILE_PINS.get(w, [paths.BF16])]
        elif repo := paths.hf_repo(w):
            new = [(repo[0], repo[1] or "main", "check")]
        else:
            sys.exit(f"{opt}: a profile ({', '.join(PROFILE_PINS)}) or ORG/NAME[@REVISION], not {w!r}")
        pins += [p for p in new if p not in pins]
    if not args.no_draft:
        pins.append((*paths.DFLASH2, None))

    todo, total = [], 0
    for repo, rev, check in pins:
        base = paths.HF_HUB / f"models--{repo.replace('/', '--')}"
        try:
            commit = commit_of(repo, rev)
            listing = files_of(repo, commit)
        except OSError as e:  # (offline: a snapshot already here is used as it is)
            try:
                snap = paths.repo_snapshot(repo, rev)
            except FileNotFoundError:
                sys.exit(f"{repo}: cannot reach {HF} ({e})")
            print(f"{repo}: cannot reach {HF} ({e}); using {snap}", flush=True)
            continue
        snap = base / "snapshots" / commit
        missing = [(p, s, h) for p, s, h in listing if not ((snap / p).is_file() and (snap / p).stat().st_size == s)]
        if check and missing and not args.no_check:
            why = check_remote(repo, commit, listing)
            if why and not why.startswith("skipped"):
                sys.exit(f"{repo}@{commit[:7]} cannot be served: {why}")
            print(f"{repo}@{commit[:7]}: " + (why or "the engine accepts it"), flush=True)
        if rev != commit:  # (the branch or tag, for paths.repo_snapshot)
            (base / "refs").mkdir(parents=True, exist_ok=True)
            (base / "refs" / rev).write_text(commit, encoding="utf-8")
        print(f"{repo}@{commit[:7]}: {len(missing)} files to download -> {snap}", flush=True)
        for p, s, h in missing:
            (snap / p).parent.mkdir(parents=True, exist_ok=True)
            todo.append((repo, commit, p, s, h, snap / p))
            part = snap / (p + ".incomplete")
            total += s - (part.stat().st_size if part.exists() else 0)
    if not todo:
        print("all checkpoints are present")
        return
    place = paths.HF_HUB.absolute()
    while not place.exists():  # (the first start: the drive it will be on)
        place = place.parent
    free = shutil.disk_usage(place).free
    if free < total + (1 << 30):
        sys.exit(f"not enough disk space: {total / 1e9:.1f} GB to download, {free / 1e9:.1f} GB free")
    print(f"downloading {total / 1e9:.1f} GB", flush=True)
    progress = Progress(total)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futures = [ex.submit(fetch_retrying, *t, progress) for t in todo]
        errors = [f.exception() for f in futures if f.exception()]
    if errors:
        sys.exit("download failed: " + "; ".join(map(str, errors)))
    print(f"done in {time.time() - progress.t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
