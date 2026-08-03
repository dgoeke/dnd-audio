{
  description = "dnd-audio — local audio ingestion and transcription for tabletop sessions";

  # Pinned to the same channel the target host's NixOS configuration tracks, so the
  # repository shell and the host do not drift apart. See ADR-0002. Bumping this is
  # its own commit with its own gate run: it can move Python's patch version and
  # every native library underneath the wheels.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs =
    { self, nixpkgs }:
    let
      # The target host is the only host this project runs on, and `buildFHSEnv` is
      # Linux-only. Claiming systems that are never built or tested would be a
      # promise the flake cannot keep.
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = nixpkgs.lib;

      python = pkgs.python312;

      # Native libraries the CPU wheels link against at import time. manylinux
      # policy treats libstdc++ and zlib as system libraries, so wheels do not
      # bundle them and a Nix interpreter will not find them without help.
      nativeLibs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ];

      # Some wheels have no binary distribution for this platform and build on
      # install; some C extensions are compiled by uv from an sdist.
      buildTools = with pkgs; [
        cmake
        gnumake
        ninja
        pkg-config
      ];

      # Two gfx1151 knobs, applied by both shells. Neither is tuning-for-taste and both
      # fail *silently* when unset, which is why they are set here rather than left in a
      # README (`dnd-audio doctor` re-checks them, so this stays verifiable):
      #
      #   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL — gfx1151 is not on AOTriton's
      #     officially supported list, so its SDPA kernels are gated behind this flag.
      #     Without it Torch falls back to the math SDPA backend: correct, much slower,
      #     and nothing in the output says the fast path was skipped.
      #   HSA_ENABLE_SDMA — stability, not speed. gfx1151's SDMA copy engines are
      #     implicated in ring timeouts and GPU resets during large transfers. With SDMA
      #     off, copies go through compute-queue blits, which on a UMA host costs
      #     approximately nothing.
      #
      # Deliberately NOT promoted to host defaults yet; that waits for M6b's real
      # transcription smoke test. ComfyUI's other knobs (HSA_USE_SVM, MIOPEN_FIND_MODE)
      # are separate performance tuning and are not assumed to help ASR.
      rocmEnv = {
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1";
        HSA_ENABLE_SDMA = "0";
      };

      exportRocmEnv = lib.concatStringsSep "\n" (
        lib.mapAttrsToList (name: value: "export ${name}=${value}") rocmEnv
      );

      # AMD's gfx1151 Torch wheels pull a `rocm[libraries]` sdist that builds at install
      # time and expects a /usr/lib layout, which is what an FHS sandbox provides and
      # `mkShell` does not.
      #
      # M0 proved only that this opens; M6a used it in anger and the package list below
      # needed no additions — the `rocm` sdist built first time against exactly this
      # toolchain (ADR-0025). It is now a tested set rather than a starting point.
      fhsEnv = pkgs.buildFHSEnv {
        name = "dnd-audio-fhs";
        targetPkgs =
          p:
          (with p; [
            python312
            uv

            # Toolchain — the ROCm sdist compiles at install time.
            gcc
            cmake
            gnumake
            ninja
            pkg-config

            stdenv.cc.cc.lib
            zlib
            openssl

            ffmpeg
            sox

            git
            curl
            which
          ]);
        profile = exportRocmEnv;
        runScript = "bash";
      };
    in
    {
      devShells.${system} = {
        # The everyday shell. direnv activates it via `.envrc` on `cd`, so an agent
        # or developer who forgets is corrected by the environment rather than by a
        # confusing test failure. Everything through M5 works here; nothing in this
        # project requires the FHS shell to run the gate.
        default = pkgs.mkShell {
          name = "dnd-audio";

          packages = [
            python
            pkgs.uv
            pkgs.ffmpeg # carries libmp3lame, loudnorm, and ebur128 — M5 needs all three
            pkgs.sox
          ]
          ++ buildTools
          ++ nativeLibs;

          shellHook = ''
            # uv manages the project's .venv, but it must never choose the
            # interpreter itself: the host's own python3 is 3.13, which
            # `requires-python` excludes, and uv would otherwise happily download a
            # managed CPython from the network to satisfy a constraint the flake
            # already satisfies.
            export UV_PYTHON=${python}/bin/python3.12
            export UV_PYTHON_DOWNLOADS=never

            export LD_LIBRARY_PATH=${lib.makeLibraryPath nativeLibs}''${LD_LIBRARY_PATH:+:}''${LD_LIBRARY_PATH:-}

            # Set here too, not only in the FHS shell. `dnd-audio doctor` runs from this
            # shell and reports on them, and a variable that were set only where the GPU
            # work happens would make doctor's answer differ from the run's.
            ${exportRocmEnv}
          '';
        };

        # Entered explicitly: `nix develop .#fhs`. Deliberately NOT direnv-activated
        # — `nix print-dev-env` ends by evaluating the shellHook, and this one execs
        # into bwrap, so sourcing it would replace direnv's evaluation shell. That
        # was tested; see ADR-0002 before revisiting it.
        #
        # That same exec is why `nix develop .#fhs --command CMD` silently runs
        # nothing: the hook replaces the shell before CMD is reached. Use an
        # interactive session, or `nix run .#fhs -- -c 'CMD'` (below) to script it.
        fhs = fhsEnv.env;
      };

      # The same FHS sandbox as a runnable wrapper: `nix run .#fhs -- -c 'CMD'`.
      # `devShells.fhs` cannot be driven non-interactively, and M6a needs a way to
      # run an install or a smoke test inside the sandbox from a script.
      packages.${system}.fhs = fhsEnv;
    };
}
