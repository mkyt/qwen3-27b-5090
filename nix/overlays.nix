# Overlays of the flake. `default` adds the qwen27b-* packages to a nixpkgs
# (the flake's own package set is built from it); it does not change any
# package of the nixpkgs it is applied to.
{ self, lib }:
let
  mkDate =
    longDate:
    lib.concatStringsSep "-" [
      (builtins.substring 0 4 longDate)
      (builtins.substring 4 2 longDate)
      (builtins.substring 6 2 longDate)
    ];
  version = "0.1.0+date=${mkDate (self.lastModifiedDate or "19700101")}_${self.shortRev or "dirty"}";
in
{
  default = self.overlays.qwen27b;

  qwen27b = final: prev: {
    # A private Python 3.12 with the overrides the server and the tests need.
    # Being its own interpreter (`self` points back at it, so withPackages and
    # .pkgs see the overrides), it leaves python312 of the nixpkgs alone.
    qwen27b-python = prev.python312.override {
      self = final.qwen27b-python;
      packageOverrides = pyfinal: pyprev: {
        # anyio (via httpx in the dev shell) installs fine; only its own test
        # suite fails on the build machines under Python 3.12 (TLS / asyncio
        # threading tests).
        anyio = pyprev.anyio.overridePythonAttrs { doCheck = false; };
        # The test inputs of llguidance (torch, transformers) and safetensors
        # (torch) are the source-built CUDA PyTorch, which no binary cache has:
        # running those tests compiles PyTorch, NCCL and NVSHMEM.
        llguidance = pyprev.llguidance.overridePythonAttrs { doCheck = false; };
        safetensors = pyprev.safetensors.overridePythonAttrs { doCheck = false; };
        # Pillow 12.3.0, the version frontend/requirements.txt pins:
        # frontend/imageproc.py resizes bit-exactly like Pillow's bicubic resize
        # (tests/test_imageproc.py), and Pillow 13 rounds some pixels otherwise.
        pillow = pyprev.pillow.overridePythonAttrs {
          version = "12.3.0";
          src = final.fetchFromGitHub {
            owner = "python-pillow";
            repo = "Pillow";
            tag = "12.3.0";
            hash = "sha256-kmUlgR+f75Y8DAKKPdEbchLLgg0m95oyVP53WTQni88=";
          };
        };
      };
    };

    # A fixed version: one with the commit in it would change the store path,
    # and rebuild the CUDA engine, on every commit, not only on engine changes.
    qwen27b-engine = final.callPackage ./engine.nix {
      version = "0.1.0";
      # CUDA 13.3, as the dev and serving images (nvidia/cuda:13.3.0-devel).
      cudaPackages = final.cudaPackages_13_3;
    };

    qwen27b-serve = final.callPackage ./serve.nix { python = final.qwen27b-python; };

    qwen27b-runtime-files = final.callPackage ./runtime-files.nix { inherit version; };
  };
}
