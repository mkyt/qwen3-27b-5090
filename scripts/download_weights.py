"""Download the pinned serving checkpoints (gdnfp4 and the DFlash2 draft, 23 GB) into the Hugging Face cache, with
the Python standard library only: the Windows package runs it before the server (no huggingface_hub, uv or symlinks).

    python scripts/download_weights.py [--names GDNFP4,DFLASH2] [--jobs 4]

The files go to $HF_HOME/hub/models--<org>--<name>/snapshots/<revision>/ (HF_HOME defaults to ~/.cache/huggingface),
where reference/qwen27b/paths.py looks for them. Files already there with the right size are kept; a download resumes
from its .incomplete file, and every LFS file is checked against its SHA-256 before it takes its final name.
HF_TOKEN, if set, is sent (none of the repositories is gated; it only raises the rate limit).
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from qwen27b import paths  # noqa: E402

HF = "https://huggingface.co"
CHUNK = 8 << 20


def request(url, headers=None):
    h = {"User-Agent": "qwen3-27b-5090-download"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return urllib.request.Request(url, headers={**h, **(headers or {})})


def files_of(repo, rev):
    """The files of a revision: [(path, size, sha256 or None)]."""
    url = f"{HF}/api/models/{repo}/tree/{rev}?recursive=true"
    with urllib.request.urlopen(request(url), timeout=60) as r:
        tree = json.load(r)
    return [(t["path"], t["size"], t.get("lfs", {}).get("oid")) for t in tree if t["type"] == "file"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="GDNFP4,DFLASH2", help="pins in reference/qwen27b/paths.py")
    ap.add_argument("--jobs", type=int, default=4, help="files downloaded at once")
    args = ap.parse_args()

    todo, total = [], 0
    for name in args.names.split(","):
        repo, rev = getattr(paths, name)
        snap = paths.HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
        try:
            listing = files_of(repo, rev)
        except OSError as e:  # (offline: a snapshot already here is used as it is)
            if (snap / "config.json").is_file():
                print(f"{repo}: cannot reach {HF} ({e}); using {snap}", flush=True)
                continue
            sys.exit(f"{repo}: cannot reach {HF} ({e})")
        missing = [(p, s, h) for p, s, h in listing if not ((snap / p).is_file() and (snap / p).stat().st_size == s)]
        print(f"{repo}@{rev[:7]}: {len(missing)} files to download -> {snap}", flush=True)
        for p, s, h in missing:
            (snap / p).parent.mkdir(parents=True, exist_ok=True)
            todo.append((repo, rev, p, s, h, snap / p))
            part = snap / (p + ".incomplete")
            total += s - (part.stat().st_size if part.exists() else 0)
    if not todo:
        print("all checkpoints are present")
        return
    free = shutil.disk_usage(paths.HF_HUB.parent if paths.HF_HUB.parent.exists() else pathlib.Path.home()).free
    if free < total + (1 << 30):
        sys.exit(f"not enough disk space: {total / 1e9:.1f} GB to download, {free / 1e9:.1f} GB free")
    print(f"downloading {total / 1e9:.1f} GB", flush=True)
    progress = Progress(total)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futures = [ex.submit(fetch, *t, progress) for t in todo]
        errors = [f.exception() for f in futures if f.exception()]
    if errors:
        sys.exit("download failed: " + "; ".join(map(str, errors)))
    print(f"done in {time.time() - progress.t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
