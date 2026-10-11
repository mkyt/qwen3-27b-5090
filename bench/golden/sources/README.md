# Golden prompt sources

`reference/golden.py prompts` tokenizes these texts and keeps prefixes of 128, 512, 4,096 and 16,384 tokens.
The files are cut at a line boundary just past the longest prefix they provide.

| File | Text | Origin | License |
| --- | --- | --- | --- |
| `ja-prose.txt` | Natsume Soseki, *Kokoro* (1914), from the start | [Aozora Bunko](https://www.aozora.gr.jp/cards/000148/card773.html), file `773_ruby_5968.zip`; ruby, annotations and the header / colophon removed | Public domain |
| `en-prose.txt` | William Hewitt, *Observations on Coroners* (1852), from about a tenth into the book | [Project Gutenberg #58104](https://www.gutenberg.org/ebooks/58104); text between the START / END markers | Public domain |
| `code-python.txt` | A snapshot of `scripts/bench_agentic.py` (2026-10-11) | This repository | MIT |
| `code-cpp.txt` | A snapshot of `engine/src/model.cc` (2026-10-11) | This repository | MIT |

The code files are frozen copies, so the prompts do not change when the code does. They predate the rename of the
engine's identifiers from `q38` / `Q38_` to `q27` / `Q27_`.

The texts should not be ones the model has memorized: the golden comparison against the BF16 model then measures
how much of the recitation survives quantization rather than the quantization error. *Pride and Prejudice* was tried
first and dropped for this (the BF16 model's mean NLL was 0.2-0.4 nats/token; *Kokoro* 3.2, this book 2.3 with the
NVFP4 checkpoint).
