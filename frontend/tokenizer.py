"""The checkpoint's tokenizer and chat template without transformers: `tokenizers` for the ids and jinja2 for the
template, set up as transformers 5.18 does (`TokenizersBackend`, `render_jinja_template`), so that a prompt renders
and encodes to the same ids (frontend/tests/test_tokenizer.py compares them where transformers is installed).
"""

import hashlib
import json
import pathlib
import re
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


def _byte_decoder():
    """GPT-2's byte-level alphabet (the ByteLevel pre-tokenizer's): character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()

# Marks of client text in a rendered prompt (Tokenizer.prompt_ids): ESC + one character per added token, ESC ESC for
# an ESC the client wrote. Private-use characters, which the template passes through unchanged.
ESC = "\U0010fffd"
MARK0 = 0xF0000
RESULT = ("<tool_response>", "</tool_response>")


def fingerprint(path):
    r"""What gives a model directory's token ids their meaning (vocabulary, merges and added tokens of tokenizer.json)
    as a digest: the server keeps the tokenizer it started with, so it starts an engine on other weights (launch.py)
    only if theirs is the same. A fine-tune's tokenizer.json rewritten by another transformers differs in the rest
    (e.g. Huihui's abliterated checkpoint: merges as "a b" strings, a pre-tokenizer pattern without \p{M}, ByteLevel
    decoder flags): the server's own splitting is kept for those. The chat template is not part of it: prompts are
    rendered with that of the weights the engine runs (Tokenizer(path, template=...))."""
    path = pathlib.Path(path) / "tokenizer.json"
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _fingerprints:
        tok = json.loads(path.read_text(encoding="utf-8"))
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
        # (an export can leave these in the file, and they would cut a long prompt without an error)
        self.tok.no_truncation()
        self.tok.no_padding()
        path = self.template_path
        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = path / "chat_template.jinja"
        source = template.read_text(encoding="utf-8") if template.is_file() else config["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[GenerationTag, jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        self.template = env.from_string(source)
        # (transformers passes its special tokens map to the template; kwargs override them)
        self.special = {k: v for k, v in config.items() if k.endswith("_token") and isinstance(v, str)}
        added = self.tok.get_added_tokens_decoder()
        self.added = set(added)
        self._pieces = {}
        # Client text is kept as text (prompt_ids): the same tokenizer without added tokens encodes the parts of a
        # rendered prompt that came from the client, so "<|im_start|>" there is "<", "|", "im", ... and not the token.
        spec = json.loads((self.path / "tokenizer.json").read_text(encoding="utf-8"))
        spec.update(added_tokens=[], truncation=None, padding=None, post_processor=None)
        self.plain = RustTokenizer.from_str(json.dumps(spec))
        strings = sorted({t.content for t in added.values()}, key=lambda s: (-len(s), s))
        self._added_id = {t.content: i for i, t in added.items()}
        self._mark = {s: ESC + chr(MARK0 + k) for k, s in enumerate(strings)}
        self._unmark = {m[1]: s for s, m in self._mark.items()}
        self._unmark[ESC] = ESC
        self._added_re = re.compile("|".join(map(re.escape, strings)) or "(?!)")
        self._marked_re = re.compile(ESC + "(.)", re.S)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)

    def token_id(self, token):
        return self.tok.token_to_id(token)

    def piece(self, i):
        """(text, UTF-8 bytes) of token i alone (logprobs): its bytes in the byte-level vocabulary, which may end inside a
        character (the text then has U+FFFD for it, as a decode of the token alone does); an added token's own text."""
        got = self._pieces.get(i)
        if got is None:
            t = self.tok.id_to_token(i) or ""
            if i in self.added or any(c not in BYTE_DECODER for c in t):
                b = t.encode("utf-8")
            else:
                b = bytes(BYTE_DECODER[c] for c in t)
            got = self._pieces[i] = (b.decode("utf-8", errors="replace"), b)
        return got

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        """The prompt text of chat messages (transformers' apply_chat_template with tokenize=False)."""
        return self.template.render(**{**self.special, "messages": messages, "tools": tools, "documents": None,
                                       "add_generation_prompt": add_generation_prompt, **kwargs})

    def prompt_ids(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        """The prompt ids of chat messages, with what the client wrote kept as text: an added token's string
        ("<|im_start|>", "</tool_response>", "<tool_call>", "<|image_pad|>", ...) in a system, user or tool message or
        in the tool definitions encodes as ordinary tokens, so it cannot end a turn, open a fake one or stand for an
        image. Assistant messages (content, reasoning, tool calls) are the model's own output and keep their tokens.
        Without such strings the ids are those of encode(render(...)), as transformers has them."""
        messages = [m if not isinstance(m, dict) or m.get("role") == "assistant" else self._escape_message(m)
                    for m in messages]
        text = self.render(messages, tools=self._escape(tools), add_generation_prompt=add_generation_prompt, **kwargs)
        return self.encode_marked(text)

    def _escape(self, x):
        """x (strings, lists and mappings of them, keys too) with the added tokens' strings marked."""
        if isinstance(x, str):
            if ESC in x:
                x = x.replace(ESC, ESC + ESC)
            return self._added_re.sub(lambda m: self._mark[m.group()], x)
        if isinstance(x, list):
            return [self._escape(v) for v in x]
        if isinstance(x, dict):
            return {self._escape(k): self._escape(v) for k, v in x.items()}
        return x

    def _escape_message(self, m):
        """A system, user or tool message with its text marked. The template takes a user message that is a whole
        "<tool_response>...</tool_response>" as a tool result (clients that send results as user turns): those two
        stay as they are."""
        content = m.get("content")
        if isinstance(content, str):
            keep = m.get("role") == "user" and self._is_result(content, content)
            content = self._wrapped(content, keep, keep)
        elif isinstance(content, list):
            def text(p):
                return p.get("text") if isinstance(p, dict) and p.get("type") == "text" else None

            ends = len(content) > 0 and text(content[0]), len(content) > 0 and text(content[-1])
            keep = m.get("role") == "user" and self._is_result(*ends)
            content = [dict(p, text=self._wrapped(text(p), keep and k == 0, keep and k == len(content) - 1))
                       if text(p) is not None else p for k, p in enumerate(content)]  # (images: their URLs)
        return dict(self._escape({k: v for k, v in m.items() if k != "content"}), content=content)

    @staticmethod
    def _is_result(first, last):
        """Whether text that starts with `first` and ends with `last` is a whole tool result (template: trimmed
        content.startswith('<tool_response>') and .endswith('</tool_response>'))."""
        return (isinstance(first, str) and isinstance(last, str) and first.lstrip().startswith(RESULT[0])
                and last.rstrip().endswith(RESULT[1]))

    def _wrapped(self, text, keep_start, keep_end):
        """text marked, but for a leading "<tool_response>" (keep_start) and a trailing "</tool_response>" (keep_end)."""
        if not isinstance(text, str):
            return self._escape(text)
        a, b = 0, len(text)
        if keep_start:
            a = len(text) - len(text.lstrip()) + len(RESULT[0])
        if keep_end:
            b = max(a, len(text.rstrip()) - len(RESULT[1]))
        return text[:a] + self._escape(text[a:b]) + text[b:]

    def encode_marked(self, text):
        """Ids of a rendered prompt whose client text was marked (_escape): the added tokens the template wrote are
        tokens, the text between them is encoded without added tokens, with the marked strings put back."""
        if ESC not in text:
            return self.encode(text)
        out, parts, at = [], [], 0
        for m in self._added_re.finditer(text):
            parts.append((len(out), text[at:m.start()]))
            out.append(self._added_id[m.group()])
            at = m.end()
        parts.append((len(out), text[at:]))
        unmark = lambda m: self._unmark[m.group(1)]  # noqa: E731
        segs = [self._marked_re.sub(unmark, s) for _, s in parts]
        encs = self.plain.encode_batch(segs, add_special_tokens=False)
        for (pos, _), e in reversed(list(zip(parts, encs))):
            out[pos:pos] = e.ids
        return out
