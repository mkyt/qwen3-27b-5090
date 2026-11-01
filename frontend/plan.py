"""What POST /admin/engine/plan answers, without a server: for a Hugging Face cache the running server does not use, or
with no server running (the tray app's settings window). HF_HOME says the cache, and the variables the settings as the
server would start with them. Q27_PLAN_REQUEST has the same JSON as the POST ({"settings": {...}}), and optionally
"device": [free, total] bytes (default: NVML, as the server with no engine). Prints the answer, with "draft" (whether the
DFlash2 draft is in the cache) and "links" (paths.unreadable_repos), or {"error": ...}. Needs no GPU and allocates nothing.
"""

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "reference"), str(ROOT / "engine/python"), str(ROOT / "frontend")]

from launch import Launch, env_settings, set_engine_env  # noqa: E402
from qwen27b import paths  # noqa: E402
from stats import HUB, Nvml  # noqa: E402


def has_draft():
    try:
        return (paths.snapshot(*paths.DFLASH2) / "config.json").is_file()
    except FileNotFoundError:
        return False


def main():
    request = json.loads(os.environ.get("Q27_PLAN_REQUEST") or "{}")
    set_engine_env(ROOT)
    if not request.get("device"):
        try:
            HUB.nvml = Nvml()
        except OSError:
            pass
    state = {"draft": has_draft(), "links": paths.unreadable_repos()}
    device, source = (tuple(request["device"]), "server") if request.get("device") else HUB.plan_device()
    if device is None:
        return {**state, "error": "the device's memory is unknown (no NVML)"}
    launch = Launch({"prefill_chunk": int(os.environ.get("PREFILL_CHUNK", 2048)), "engine_python": str(ROOT / "engine/python")},
                    env_settings())
    changes = dict(request.get("settings") or {})
    missing = changes.pop("weights") if changes.get("weights") not in (None, *launch.weights_list()) else None
    try:
        answer = {**state, **launch.plan_answer(changes, device, source)}
        if missing:  # (the other choices still have their estimates)
            answer["plan"] = {"error": f"{missing} is not in this folder"}
        return answer
    except ValueError as e:
        return {**state, "weights": launch.weights_choices(), "error": str(e)}


if __name__ == "__main__":
    print(json.dumps(main()))
