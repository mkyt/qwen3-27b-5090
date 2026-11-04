# Tests that run without a GPU or the weights (`nix flake check`).
{
  stdenvNoCC,
  python,
}:
let
  # The whole checkout: the tests read compose.yaml, docs/, scripts/ and bench/ too.
  mkTest =
    {
      name,
      packages,
      command,
    }:
    stdenvNoCC.mkDerivation {
      inherit name;
      src = ../.;
      nativeBuildInputs = [ (python.withPackages packages) ];
      dontConfigure = true;
      dontBuild = true;
      doCheck = true;
      checkPhase = command;
      installPhase = "touch $out";
    };
in
{
  # The server's parsing, sampling, structured output, admission and image /
  # video preprocessing, with the server's packages only (the tests' torch and
  # checkpoint parts skip).
  frontend-tests = mkTest {
    name = "qwen27b-frontend-tests";
    packages =
      ps: with ps; [
        jinja2
        llguidance
        numpy # imported by tests/test_tokenizer.py and tests/test_videoproc.py
        pillow
        tokenizers
        pytest
      ];
    command = "cd frontend && python -m pytest -q tests/";
  };

  # The reference model's GDN math and quantization on the CPU (the snapshot
  # tests need the checkpoint).
  reference-tests = mkTest {
    name = "qwen27b-reference-tests";
    packages =
      ps: with ps; [
        torch-bin
        transformers
        safetensors
        numpy
        pillow
        pytest
      ];
    command = "cd reference && PYTHONPATH=. python -m pytest -q tests/test_gdn.py tests/test_quant.py";
  };
}
