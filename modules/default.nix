# services.qwen27b -- run the qwen3-27b-5090 engine (libq27.so + the frontend
# API server) as a systemd service.
#
# The module is agnostic of the engine build / python environment / runtime
# tree: the caller provides them via the `engine`, `serve` and `files`
# options. Defaults follow the upstream README's "Configuration" table.
#
# The engine only runs on an RTX 5090 (SM120 kernels); the service fails
# prestart with a clear message on any other card or when the NVIDIA driver
# is not enabled.

{ config, lib, pkgs, ... }:

let
  inherit (lib) mkEnableOption mkOption types;

  cfg = config.services.qwen27b;

  # CUDA 13.3, matching the engine's dev/Dockerfile (nvidia/cuda:13.3.0-devel).
  cuda = pkgs.cudaPackages_13_3;
in
{
  options.services.qwen27b = {
    enable = mkEnableOption "the qwen3-27b-5090 native engine as a systemd service";

    # --- The three upstream flake packages ---
    engine = mkOption {
      type = types.package;
      description = ''
        Derivation installing libq27.so under $out/lib
        (e.g. `qwen27b.packages."x86_64-linux".engine`).
      '';
    };
    serve = mkOption {
      type = types.package;
      description = ''
        Python environment with torch, transformers, safetensors, numpy and
        pillow (e.g. `qwen27b.packages."x86_64-linux".serve`).
      '';
    };
    files = mkOption {
      type = types.package;
      description = ''
        Derivation providing the runtime tree under $out: frontend/,
        engine/python/, reference/, bench/draft-vocab/vocab-128k.txt
        (e.g. `qwen27b.packages."x86_64-linux".repoFiles`).
      '';
    };

    # Where the gdnfp4 checkpoint + DFlash2 draft live (the server reads them
    # from the HF cache; HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE are set below).
    modelPath = mkOption {
      type = types.str;
      # Default to the service user's home (a systemd StateDirectory that is
      # created and owned by the user — unlike the upstream docker-compose
      # default `./hf`, which under systemd resolves to `/hf`, owned by root).
      # Override to point at an HF cache you already populated (e.g. with
      # `scripts/download_weights.py`, see the README).
      default = "/var/lib/qwen27b/hf";
      description = ''
        HF home for the weights (`HF_HOME`). `HF_HUB_OFFLINE` +
        `TRANSFORMERS_OFFLINE` are set, so this must already contain the model.
      '';
    };

    # --- README "Configuration" defaults ---
    # The serve service reads its settings from the environment (or .env);
    # the most useful ones:
    maxTokens = mkOption {
      type = types.int;
      default = 262144;
      description = "Context limit of one request (MAX_TOKENS).";
    };
    maxSeqs = mkOption {
      type = types.int;
      default = 8;
      description = ''
        Requests that can run at once (1-8) (MAX_SEQS); how many actually run
        is decided while running, from the free KV cache.
      '';
    };
    kvDtype = mkOption {
      type = types.enum [ "nvfp4" "fp8" "bf16" ];
      default = "nvfp4";
      description = "KV cache dtype (KV_DTYPE).";
    };
    weights = mkOption {
      type = types.str;
      default = "gdnfp4";
      description = ''
        Weight checkpoint (WEIGHTS): gdnfp4, nvidia, quasar, or a model
        directory.
      '';
    };
    spec = mkOption {
      type = types.enum [ "dflash2" "none" ];
      default = "dflash2";
      description = "Speculative decode (SPEC): dflash2 or none.";
    };
    vision = mkOption {
      type = types.enum [ "host" "fp8" "bf16" "off" ];
      default = "host";
      description = "Vision tower placement (VISION): host, fp8, bf16, or off.";
    };
    draftQuant = mkOption {
      type = types.nullOr (types.enum [ "fp4" "fp8" "bf16" ]);
      default = "fp4";
      description = ''
        Draft (DFlash2) linear-layer precision (Q27_DRAFT_QUANT). `fp4` (NVFP4)
        is the documented default: same speculative acceptance as `bf16` but
        the fastest decode. Set to `null` to leave the variable unset (the
        engine then uses its built-in default).
      '';
    };
    servedModelName = mkOption {
      type = types.str;
      default = "qwen3.8-27b";
      description = "The served model name (SERVED_MODEL_NAME) in the OpenAI/Anthropic API.";
    };
    stateHostCacheGiB = mkOption {
      type = types.int;
      default = 4;
      description = ''
        GiB of pinned host RAM for the prefix cache (STATE_HOST_CACHE) -- GDN
        state checkpoints that let a growing conversation / a repeated long
        document restore instead of re-prefilling. 0 disables.
      '';
    };
    checkpointInterval = mkOption {
      type = types.int;
      default = 8192;
      description = "Prefix-cache checkpoint stride in tokens (CHECKPOINT_INTERVAL).";
    };
    kvPoolTokens = mkOption {
      type = types.int;
      default = 0;
      description = ''
        Fixed GPU KV pool size (KV_POOL_TOKENS). 0 = auto-fill against the live
        VRAM budget at boot (the engine's default).
      '';
    };

    port = mkOption {
      type = types.int;
      default = 8000;
      description = "Loopback port the API server binds.";
    };
    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address the API server binds.";
    };

    user = mkOption {
      type = types.str;
      default = "qwen27b";
      description = "System user the service runs as.";
    };

    createUser = mkOption {
      type = types.bool;
      default = true;
      description = ''
        Whether the module creates `user` (system user + group) itself. Set to
        false to run the service as an existing user you manage elsewhere (e.g.
        a shared inference user) — the module then only declares the service
        and its `StateDirectory`, and leaves the user/group to you.
      '';
    };

    userHome = mkOption {
      type = types.str;
      default = "/var/lib/qwen27b";
      description = ''
        Home directory of the service user (doubles as `$HOME`). Created by
        systemd's `StateDirectory` and owned by the service user, so it exists
        even though a system user has `createHome = false`.
      '';
    };

    gpuLockFile = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = ''
        Path to the GPU lease (flock) file. When set, `ARCANE_GPU_LOCK` is
        exported so a flock-wrapping launcher (e.g. the machine's
        gpu_launch.py) can acquire exclusive GPU ownership before the server
        loads CUDA weights. The launcher is supplied via `execStart`.
      '';
    };

    extraEnv = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Additional environment variables for the service.";
    };

    extraPath = mkOption {
      type = types.listOf types.package;
      default = [ ];
      description = ''
        Extra packages added to the unit's PATH (on top of bash + the serve
        environment). For example the interpreter that runs a flock-wrapping
        launcher.
      '';
    };

    # Full replacement for the server invocation (the systemd ExecStart). When
    # null, the module builds the default `server.py --host .. --port .. --..`
    # command from the options above. The machine wrapper overrides this to run
    # the server through the gpu_launch.py flock lease.
    execStart = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = ''
        Optional full ExecStart command. When null, the module uses the
        default `server.py` invocation derived from host/port/maxTokens/
        kvDtype/spec.
      '';
    };

    # --- systemd knobs (rarely overridden) ---
    wantedBy = mkOption {
      type = types.listOf types.str;
      default = [ "multi-user.target" ];
      description = "systemd units that want this service.";
    };
    after = mkOption {
      type = types.listOf types.str;
      default = [ "network.target" ];
      description = "systemd units this service starts after.";
    };
    requires = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "systemd units this service requires.";
    };
    conflicts = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "systemd units this service conflicts with (one engine owns the GPU at a time).";
    };
    serviceConfig = mkOption {
      type = types.attrs;
      default = {
        Type = "simple";
        Restart = "on-failure";
        RestartSec = "10s";
        TimeoutStopSec = "120s";
      };
      description = "Additional systemd serviceConfig entries (Type, Restart, ...).";
    };
  };

  config = lib.mkIf cfg.enable {

    # The service runs as its own system user so the weights/cache live in a
    # private home, isolated from the rest of the system. NVIDIA device nodes
    # are world-writable (mode 666) once the driver is enabled, so no extra
    # group membership is required for /dev/nvidia*. When `createUser` is false
    # the caller owns the user (e.g. a shared inference account) and the module
    # must not touch `users.*`.
    users = lib.mkIf cfg.createUser {
      users.${cfg.user} = {
        isSystemUser = true;
        group = cfg.user;
        home = cfg.userHome;
        useDefaultShell = true;
      };
      groups.${cfg.user} = { };
    };

    systemd.services.qwen27b =
      let
        # Default server invocation; the wrapper can replace via `execStart`.
        defaultExecStart = ''
          ${cfg.serve}/bin/python3 ${cfg.files}/frontend/server.py \
            --host ${cfg.host} --port ${toString cfg.port} \
            --max-tokens ${toString cfg.maxTokens} \
            --kv-dtype ${cfg.kvDtype} --spec ${cfg.spec}
        '';
        execStart = cfg.execStart or defaultExecStart;
      in {
      description = "qwen3-27b-5090 native engine (libq27.so + frontend API server)";
      after = cfg.after;
      requires = cfg.requires;
      conflicts = cfg.conflicts;
      wantedBy = cfg.wantedBy;

      environment = {
        # The server (frontend/server.py) reads these from env.
        SERVED_MODEL_NAME = cfg.servedModelName;
        WEIGHTS = cfg.weights;
        KV_DTYPE = cfg.kvDtype;
        SPEC = cfg.spec;
        VISION = cfg.vision;
        # The service user has createHome=false, so systemd does not set
        # $HOME; the HF cache under HF_HOME and any `~`-relative paths in the
        # server need a real, user-owned home.
        HOME = cfg.userHome;
        STATE_HOST_CACHE = toString cfg.stateHostCacheGiB;
        CHECKPOINT_INTERVAL = toString cfg.checkpointInterval;
        MAX_TOKENS = toString cfg.maxTokens;
        MAX_TOKENS_FIT = "1";
        MAX_SEQS = toString cfg.maxSeqs;
        KV_POOL_TOKENS = toString cfg.kvPoolTokens;
        HF_HOME = cfg.modelPath;
        HF_HUB_OFFLINE = "1";
        TRANSFORMERS_OFFLINE = "1";
        PYTHONPATH = lib.concatStringsSep ":" [
          "${cfg.files}/frontend"
          "${cfg.files}/engine/python"
          "${cfg.files}/reference"
        ];
        LD_LIBRARY_PATH = lib.concatStringsSep ":" [
          "${cfg.engine}/lib"
          "${cuda.cuda_cudart}/lib"
          "${config.hardware.nvidia.package}/lib"
        ];
        Q27_LIB = "${cfg.engine}/lib/libq27.so";
      } // (lib.optionalAttrs (cfg.draftQuant != null) {
        Q27_DRAFT_QUANT = cfg.draftQuant;
      }) // (lib.optionalAttrs (cfg.gpuLockFile != null) {
        ARCANE_GPU_LOCK = cfg.gpuLockFile;
      }) // cfg.extraEnv;

      # nvidia-smi lives in the driver's separate `bin` output; include it
      # explicitly because systemd services do not inherit the interactive
      # system PATH.
      path = [ pkgs.bash cfg.serve config.hardware.nvidia.package.bin ] ++ cfg.extraPath;

      serviceConfig = cfg.serviceConfig // {
        User = cfg.user;
        # Create + own the service user's home (users.users.<n>.createHome
        # defaults to false for system users, so systemd must do it). Derive it
        # from userHome rather than user so overriding one option never leaves
        # $HOME and the created directory out of sync. `StateDirectory` is
        # relative to /var/lib, so the default /var/lib/qwen27b becomes
        # `qwen27b` (a home outside /var/lib passes through unchanged, which
        # systemd accepts from v235). systemd creates it at 0700 owned by the
        # service user.
        StateDirectory = lib.removePrefix "/var/lib/" cfg.userHome;
        # The engine only runs on an RTX 5090 (SM120); fail prestart with a
        # clear message on any other card or without the NVIDIA driver.
        # ExecStartPre is a list of executable paths. Use a generated shell
        # script so systemd executes it correctly (a raw multiline string in a
        # list is interpreted as an executable path and fails with 203/EXEC).
        ExecStartPre = [ (pkgs.writeShellScript "qwen27b-gpu-check" ''
          command -v nvidia-smi >/dev/null 2>&1 || {
            echo "services.qwen27b: ERROR: nvidia-smi not found; enable hardware.nvidia" >&2
            exit 1
          }
          GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
          case "$GPU_MODEL" in
            *RTX*5090*) : ;;
            *) echo "services.qwen27b: ERROR: this engine requires an RTX 5090 (got: $GPU_MODEL)" >&2; exit 1 ;;
          esac
        '') ];
        ExecStart = execStart;
      };
    };
  };
}
