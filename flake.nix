{
  description = "Qwen3.8-27B inference engine for one RTX 5090 (SM120)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      # The kernels are sm_120a only, and the engine is built for Linux on x86_64.
      eachSystem = lib.genAttrs [ "x86_64-linux" ];
      pkgsFor = eachSystem (
        system:
        import nixpkgs {
          localSystem = system;
          config.allowUnfree = true; # CUDA
          overlays = [ self.overlays.default ];
        }
      );
    in
    {
      overlays = import ./nix/overlays.nix { inherit self lib; };

      packages = eachSystem (
        system:
        let
          pkgs = pkgsFor.${system};
        in
        {
          default = self.packages.${system}.engine;
          engine = pkgs.qwen27b-engine;
          serve = pkgs.qwen27b-serve;
          repoFiles = pkgs.qwen27b-runtime-files;
        }
      );

      checks = eachSystem (
        system:
        self.packages.${system}
        // import ./nix/tests.nix {
          inherit (pkgsFor.${system}) stdenvNoCC;
          python = pkgsFor.${system}.qwen27b-python;
        }
      );

      # The engine's build inputs, and the Python of the reference model and the
      # kernel tests (engine/tests, reference/tests).
      devShells = eachSystem (
        system:
        let
          pkgs = pkgsFor.${system};
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ pkgs.qwen27b-engine ];
            packages = [
              (pkgs.qwen27b-python.withPackages (
                ps: with ps; [
                  torch-bin
                  torchvision-bin
                  transformers
                  safetensors
                  numpy
                  pillow
                  httpx
                  pytest
                ]
              ))
            ];
          };
        }
      );

      formatter = eachSystem (system: pkgsFor.${system}.nixfmt-tree);

      # services.qwen27b; see the README's NixOS section.
      nixosModules = {
        default = import ./nix/module.nix self;
        qwen27b = self.nixosModules.default;
      };
    };
}
