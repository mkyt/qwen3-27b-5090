"""Compile the real JSON / safetensors readers without CUDA; malformed files must fail before any data read."""

import json
import os
import pathlib
import shutil
import struct
import subprocess

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="session")
def reader(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("a C++20 compiler is needed")
    out = tmp_path_factory.mktemp("reader")
    source = out / "read.cc"
    source.write_text(r'''
#include <iostream>
#include <stdexcept>
#include "safetensors.h"
namespace q27 {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }
}
int main(int argc, char** argv) {
  try {
    q27::SafeTensors file(argv[1]);
    const auto& t = file.info("sample");
    if (argc > 2) {
      // A bounded destination: invalid declared sizes must be rejected before writing here.
      char data[16]{};
      if (t.nbytes > 16) throw std::runtime_error("test data too large");
      file.read("sample", data);
      std::cout.write(data, t.nbytes);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return 2;
  }
}
''')
    binary = out / "reader"
    flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if os.environ.get("Q27_TEST_SANITIZERS") else []
    subprocess.run([compiler, "-std=c++20", "-g", *flags, "-I", str(SRC), str(source), str(SRC / "safetensors.cc"),
                    str(SRC / "json.cc"), "-o", str(binary)], check=True, capture_output=True)
    return binary


def checkpoint(tmp_path, header=None, data=b"abcd", raw=None, index=None):
    (tmp_path / "config.json").write_text("{}")
    header = header or {"sample": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    if raw is None:
        body = json.dumps(header).encode()
        raw = struct.pack("<Q", len(body)) + body + data
    (tmp_path / "model.safetensors").write_bytes(raw)
    if index is not None:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


def run(reader, path, read=False):
    return subprocess.run([str(reader), str(path), *(["read"] if read else [])], capture_output=True, timeout=10)


def test_valid_tensor(reader, tmp_path):
    result = run(reader, checkpoint(tmp_path), read=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"abcd"


def test_header_only_plan_still_works_but_cannot_read(reader, tmp_path):
    path = checkpoint(tmp_path, data=b"")
    assert run(reader, path).returncode == 0
    result = run(reader, path, read=True)
    assert result.returncode == 2
    assert b"truncated tensor" in result.stderr


@pytest.mark.parametrize("tensor", [
    {"dtype": "BF16", "shape": [2], "data_offsets": [0, 3]},  # bf16_amax used to write 3 bytes into 2
    {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 3]},  # a matrix accepted by the old load-time quantizer
    {"dtype": "BF16", "shape": [-1], "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": [1.5], "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": [2**62, 8], "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": [0, 2**62, 8], "data_offsets": [0, 0]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [-1, 3]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [4, 0]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [0, 2**63]},
    {"dtype": "unknown", "shape": [2], "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": 2, "data_offsets": [0, 4]},
    {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4, 8]},
])
def test_invalid_tensor_metadata(reader, tmp_path, tensor):
    result = run(reader, checkpoint(tmp_path, {"sample": tensor}), read=True)
    assert result.returncode == 2, result.stderr
    assert b"Sanitizer" not in result.stderr


@pytest.mark.parametrize("raw", [b"", b"123", struct.pack("<Q", 2**63), struct.pack("<Q", 100000001),
                                struct.pack("<Q", 99) + b"{}"])
def test_invalid_header_length(reader, tmp_path, raw):
    assert run(reader, checkpoint(tmp_path, raw=raw)).returncode == 2


@pytest.mark.parametrize("header", [b'{} garbage', b'{"a":1,"a":2}', b'{"a":' + b'[' * 200 + b'0' + b']' * 200 + b'}'])
def test_malformed_json(reader, tmp_path, header):
    raw = struct.pack("<Q", len(header)) + header
    assert run(reader, checkpoint(tmp_path, raw=raw)).returncode == 2


@pytest.mark.parametrize("offsets", [[0, 4], [8, 12]])
def test_overlapping_or_gapped_tensors(reader, tmp_path, offsets):
    header = {"sample": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
              "other": {"dtype": "BF16", "shape": [2], "data_offsets": offsets}}
    assert run(reader, checkpoint(tmp_path, header, data=b"a" * 12)).returncode == 2


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/tmp/outside.safetensors", "C:\\outside.safetensors",
                                 "..\\outside.safetensors"])
def test_shard_cannot_escape_directory(reader, tmp_path, shard):
    path = checkpoint(tmp_path, index={"weight_map": {"sample": shard}})
    result = run(reader, path)
    assert result.returncode == 2
    assert b"invalid shard path" in result.stderr
