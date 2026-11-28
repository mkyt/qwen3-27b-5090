# libq27.so: the C++/CUDA engine (sm_120a; nvcc compiles it without a GPU).
{
  lib,
  stdenv,
  cmake,
  ninja,
  cudaPackages,
  version,
}:
let
  fs = lib.fileset;
in
stdenv.mkDerivation {
  pname = "qwen27b-engine";
  inherit version;

  # Only what CMake compiles, so a change elsewhere in the repository does not
  # rebuild the engine.
  src = fs.toSource {
    root = ../engine;
    fileset = fs.unions [
      ../engine/CMakeLists.txt
      ../engine/src
    ];
  };

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
    (lib.getOutput "static" cudaPackages.cuda_cudart)
  ];

  # CMakeLists.txt sets the architecture (120a), the build type (Release) and
  # C++/CUDA 20; the nixpkgs cmake hook configures, builds and installs
  # (install(TARGETS q27) puts libq27.so under $out/lib).
  # libq27.so's RUNPATH points at this cudart; the driver's libcuda (and
  # libnvcuvid for video) come from the system at run time.

  # engine/tests load libq27.so and need an RTX 5090; run them on one.
  doCheck = false;

  passthru = { inherit cudaPackages; };

  meta = {
    description = "Qwen3.8-27B inference engine (C++/CUDA, RTX 5090 / SM120)";
    homepage = "https://github.com/mkyt/qwen3-27b-5090";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
