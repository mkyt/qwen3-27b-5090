"""The built-in chat page (GET /chat, frontend/ui/chat.html): no engine needed."""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import chat  # noqa: E402
import server  # noqa: E402


def test_page_exists():
    assert server.CHAT_PAGE.is_file()


def test_no_external_resources():
    # One file that works offline: no scripts, styles, fonts or images from elsewhere.
    html = server.CHAT_PAGE.read_text(encoding="utf-8")
    assert not re.search(r"""<(script|link|img|iframe)\b[^>]*\b(src|href)\s*=\s*["']?(https?:)?//""", html, re.I)
    assert not re.search(r"@import|url\(\s*[\"']?(https?:)?//", html, re.I)


def test_same_origin_api():
    # Relative to the server's root, so the POSTs carry the server's own Origin (origin_allowed) and need no CORS.
    html = server.CHAT_PAGE.read_text(encoding="utf-8")
    for path in ("v1/chat/completions", "v1/models", "health"):
        assert f'fetch(BASE + "{path}"' in html


def test_sends_past_reasoning_back():
    # The template renders the reasoning of earlier turns; without it the prefix cache misses the last reply.
    html = server.CHAT_PAGE.read_text(encoding="utf-8")
    assert "a.reasoning_content = m.reasoning" in html


def test_thinking_options_are_the_templates_efforts():
    html = server.CHAT_PAGE.read_text(encoding="utf-8")
    select = re.search(r'<select id="think".*?</select>', html, re.S).group(0)
    options = re.findall(r'<option value="([^"]*)"', select)
    assert options[0] == "off"
    assert set(options[1:]) == set(chat.REASONING_EFFORT.values())
