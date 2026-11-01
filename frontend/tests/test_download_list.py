"""What the tray app's settings window shows for a folder: paths.unreadable_repos, download_weights.py --fix-links."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "reference")]

import download_weights as dw  # noqa: E402
from qwen27b import paths  # noqa: E402


def test_link_to_blob_keeps_the_content_and_stays_in_the_repository(tmp_path):
    base = tmp_path / "models--a--b"
    (base / "blobs").mkdir(parents=True)
    (base / "snapshots" / "c").mkdir(parents=True)
    (base / "blobs" / "x").write_text("data")
    (tmp_path / "outside").write_text("no")
    path = base / "snapshots" / "c" / "f"
    path.write_text("")  # (what stands for the LX symlink: Linux has none)
    dw.link_to_blob(path, "../../blobs/x", base)
    assert path.read_text() == "data" and path.stat().st_ino == (base / "blobs" / "x").stat().st_ino and not path.is_symlink()
    for target in ("../../../../outside", "../../blobs/missing"):
        with pytest.raises(OSError):
            dw.link_to_blob(path, target, base)
    assert path.read_text() == "data"


def test_unreadable_repos_names_the_links_that_cannot_be_followed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "HF_HUB", tmp_path)
    for repo, name in (("models--a--Qwen3.8-27B-Fine", "file"), ("models--a--Qwen3.8-27B-Linked", "link"), ("models--a--Other", "link")):
        snap = tmp_path / repo / "snapshots" / "c"
        snap.mkdir(parents=True)
        if name == "file":
            (snap / "config.json").write_text("{}")
        else:
            (snap / "config.json").symlink_to(tmp_path / "gone")
    assert paths.unreadable_repos() == ["a/Qwen3.8-27B-Linked"]
