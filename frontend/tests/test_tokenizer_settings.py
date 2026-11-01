"""A tokenizer.json can carry truncation and padding from whoever exported it: a prompt must still encode in full."""

import json
import pathlib
import sys

from tokenizers import Tokenizer as RustTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tokenizer import Tokenizer  # noqa: E402


def test_baked_truncation_and_padding_are_ignored(tmp_path):
    words = ["a", "b", "c", "d", "e"]
    tok = RustTokenizer(WordLevel({w: i for i, w in enumerate(["<unk>", "<pad>", *words])}, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.enable_truncation(max_length=3)
    tok.enable_padding(length=8, pad_id=1, pad_token="<pad>")
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": ""}), encoding="utf-8")

    assert len(RustTokenizer.from_file(str(tmp_path / "tokenizer.json")).encode("a b c d e").ids) == 8  # as exported
    assert Tokenizer(tmp_path).encode("a b c d e") == [2, 3, 4, 5, 6]
