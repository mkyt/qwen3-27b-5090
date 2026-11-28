# The files the server reads at run time, in the repository's layout
# (frontend/server.py finds them from its own path), as the serving image
# (Dockerfile) copies them: frontend/, engine/python/, the PyTorch-free
# modules of reference/qwen27b/ and the draft vocabulary.
{
  lib,
  stdenvNoCC,
  version,
}:
let
  fs = lib.fileset;
in
stdenvNoCC.mkDerivation {
  pname = "qwen27b-runtime-files";
  inherit version;

  src = fs.toSource {
    root = ../.;
    fileset = fs.difference (fs.unions [
      ../frontend
      ../engine/python
      ../reference/qwen27b/__init__.py
      ../reference/qwen27b/paths.py
      ../reference/qwen27b/image_tokens.py
      ../bench/draft-vocab/vocab-128k.txt
    ]) (fs.fileFilter (f: f.hasExt "pyc") ../.);
  };

  dontConfigure = true;
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    cp -r . $out
    runHook postInstall
  '';

  meta = {
    description = "Qwen3.8-27B API server files (frontend, engine/python, reference, draft vocabulary)";
    homepage = "https://github.com/mkyt/qwen3-27b-5090";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
