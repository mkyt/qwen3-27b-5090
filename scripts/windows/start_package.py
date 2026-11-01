"""start.cmd of the Windows package: settings.ini next to it, then the weight download and the server in this console.
The tray app (windows/tray) reads the settings the same way."""

import configparser
import os
import pathlib
import subprocess
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parents[2]


def read_settings(path):
    """The [env] section of settings.ini by upper-case name (UTF-8, UTF-16 or the ANSI code page, as Notepad saves it)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("mbcs" if os.name == "nt" else "latin-1")
    ini = configparser.ConfigParser(interpolation=None, strict=False, delimiters=("=",), comment_prefixes=(";", "#"))
    ini.optionxform = str
    ini.read_string(text)
    return {k.upper(): v.strip().strip('"') for k, v in ini["env"].items()} if ini.has_section("env") else {}


def server_env(environ, settings, root):
    """The server's environment and notes for the log: settings.ini, then the environment, then the defaults. HF_HOME
    is the exception: only settings.ini moves the weights out of this folder, not a variable of the whole PC."""
    env, notes = dict(environ), []
    outside = env.pop("HF_HOME", None)
    if outside and "HF_HOME" not in settings:
        notes.append(f"HF_HOME={outside} in the environment is not used: set it in settings.ini to use that cache")
    for k, v in settings.items():
        if k in env and env[k] != v:
            notes.append(f"{k}={v} from settings.ini (the environment has {env[k]})")
    env.update(settings)
    env.setdefault("HOST", "127.0.0.1")
    env.setdefault("PORT", "8000")
    env["HF_HOME"] = os.path.abspath(os.path.join(root, env.get("HF_HOME", "huggingface")))
    return env, notes


def main():
    env, notes = server_env(os.environ, read_settings(ROOT / "settings.ini"), ROOT)
    for note in notes:
        print(note, flush=True)
    python = sys.executable
    try:
        if subprocess.call([python, str(ROOT / "app/scripts/download_weights.py")], env=env, cwd=ROOT):
            print("The weights could not be downloaded.")
            return 1
        if env.get("OPEN_UI", "1") != "0":
            webbrowser.open(f"http://127.0.0.1:{env['PORT']}/ui")
        return subprocess.call([python, str(ROOT / "app/frontend/server.py"), "--host", env["HOST"]], env=env, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
