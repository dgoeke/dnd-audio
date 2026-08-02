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

      # Held for M6a. AMD's gfx1151 Torch wheels pull a `rocm[libraries]` sdist that
      # builds at install time and expects a /usr/lib layout, which is what an FHS
      # sandbox provides and `mkShell` does not.
      #
      # M0 proves only that this opens. The package list is modelled on the host's
      # ComfyUI service and is a starting point for M6a to refine against a real
      # ROCm install, not a validated set.
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
