"""The checkpoint's tokenizer and chat template without transformers: `tokenizers` for the ids and jinja2 for the
template, set up as transformers 5.18 does (`TokenizersBackend`, `render_jinja_template`), so that a prompt renders
and encodes to the same ids (frontend/tests/test_tokenizer.py compares them where transformers is installed).
"""

import hashlib
import json
import pathlib
from datetime import datetime

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as RustTokenizer


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # (as transformers: jinja's own tojson escapes HTML characters)
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def strftime_now(fmt):
    return datetime.now().strftime(fmt)


class GenerationTag(jinja2.ext.Extension):
    """`{% generation %}` blocks render as their body (transformers tracks them for assistant masks)."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


_fingerprints = {}


def fingerprint(path):
    """What gives a model directory's token ids their meaning (vocabulary, merges and added tokens of tokenizer.json)
    as a digest: the server keeps the tokenizer it started with, so it starts an engine on other weights (launch.py)
    only if theirs is the same. A fine-tune's tokenizer.json rewritten by another transformers differs in the rest
    (e.g. Huihui's abliterated checkpoint: merges as "a b" strings, a pre-tokenizer pattern without \p{M}, ByteLevel
    decoder flags): the server's own splitting is kept for those. The chat template is not part of it: prompts are
    rendered with that of the weights the engine runs (Tokenizer(path, template=...))."""
    path = pathlib.Path(path) / "tokenizer.json"
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _fingerprints:
        tok = json.loads(path.read_text())
        model = tok["model"]
        merges = [m if isinstance(m, str) else " ".join(m) for m in model.get("merges", ())]
        ids = [model.get("type"), model.get("vocab"), merges, tok.get("added_tokens")]
        _fingerprints[key] = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
    return _fingerprints[key]


class Tokenizer:
    """Encode / decode / render of a model directory (tokenizer.json, tokenizer_config.json, chat_template.jinja).
    template: another model directory to take the chat template and its special tokens from (weights with the same
    fingerprint that the engine runs)."""

    def __init__(self, path, template=None):
        path = pathlib.Path(path)
        self.path, self.template_path = path, pathlib.Path(template or path)
        self.tok = RustTokenizer.from_file(str(path / "tokenizer.json"))
        path = self.template_path
        config = json.loads((path / "tokenizer_config.json").read_text())
        template = path / "chat_template.jinja"
        source = template.read_text() if template.is_file() else config["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[GenerationTag, jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        self.template = env.from_string(source)
        # (transformers passes its special tokens map to the template; kwargs override them)
        self.special = {k: v for k, v in config.items() if k.endswith("_token") and isinstance(v, str)}

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_id(self, token):
        return self.tok.token_to_id(token)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        """The prompt text of chat messages (transformers' apply_chat_template with tokenize=False)."""
        return self.template.render(**{**self.special, "messages": messages, "tools": tools, "documents": None,
                                       "add_generation_prompt": add_generation_prompt, **kwargs})
