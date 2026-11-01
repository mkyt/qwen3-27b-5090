"""The settings of the Windows package (scripts/windows/start_package.py; the tray app applies the same rules)."""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts/windows"))

import start_package as sp  # noqa: E402


def test_settings_come_before_the_environment_and_the_defaults(tmp_path):
    env, notes = sp.server_env({"PORT": "8001", "HOST": "0.0.0.0", "PATH": "x"}, {"PORT": "8002"}, str(tmp_path))
    assert env["PORT"] == "8002" and env["HOST"] == "0.0.0.0" and env["PATH"] == "x"
    assert len(notes) == 1 and "PORT=8002" in notes[0] and "8001" in notes[0]
    env, notes = sp.server_env({}, {}, str(tmp_path))
    assert env["HOST"] == "127.0.0.1" and env["PORT"] == "8000" and not notes


def test_the_weights_stay_in_the_folder_unless_settings_say_otherwise(tmp_path):
    inside = os.path.join(str(tmp_path), "huggingface")
    env, notes = sp.server_env({"HF_HOME": "/elsewhere"}, {}, str(tmp_path))
    assert env["HF_HOME"] == inside and "not used" in notes[0]
    assert sp.server_env({}, {"HF_HOME": "weights"}, str(tmp_path))[0]["HF_HOME"] == os.path.join(str(tmp_path), "weights")
    assert sp.server_env({"HF_HOME": "/a"}, {"HF_HOME": "/b"}, str(tmp_path))[0]["HF_HOME"] == os.path.abspath("/b")


def test_read_settings_as_notepad_saves_them(tmp_path):
    path = tmp_path / "settings.ini"
    assert sp.read_settings(path) == {}
    text = '; comment\n[other]\nPORT=1\n[env]\n; PORT=2\nport = 8003\nHF_HOME="D:\\\u91cd\u307f"\nWDDM_MARGIN_GIB=0.5\n'
    want = {"PORT": "8003", "HF_HOME": "D:\\\u91cd\u307f", "WDDM_MARGIN_GIB": "0.5"}
    for encoded in (text.encode("utf-8"), b"\xef\xbb\xbf" + text.encode("utf-8"), text.encode("utf-16")):
        path.write_bytes(encoded)
        assert sp.read_settings(path) == want
