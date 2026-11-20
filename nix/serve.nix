# The API server's Python environment: the packages of frontend/requirements.txt
# (tokenizer.json with tokenizers, the chat template with jinja2, images with
# pillow, structured output with llguidance). No PyTorch or transformers; the
# engine runs in its own process (frontend/worker.py -> libq27.so).
{ python }:
python.withPackages (
  ps: with ps; [
    jinja2
    llguidance
    pillow
    tokenizers
  ]
)
