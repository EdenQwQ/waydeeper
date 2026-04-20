{
  description = "Waydeeper - GPU-accelerated depth effect wallpaper for Wayland (Rust)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Common arguments for crane
        commonArgs = {
          # Include .c files in the source
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type: (craneLib.filterCargoSources path type) || (builtins.match ".*\\.c$" path != null);
          };
          strictDeps = true;

          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            makeWrapper
            wayland-scanner
          ];

          buildInputs =
            with pkgs;
            [
              wayland
              wayland-protocols
              libGL
              libglvnd
              libxkbcommon
              openssl
              onnxruntime
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];

          # Set environment variables for build
          WAYLAND_SCANNER = "${pkgs.wayland}/bin/wayland-scanner";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };

        # Build dependencies only (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the actual package
        waydeeper = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;

            postInstall = ''
              wrapProgram $out/bin/waydeeper \
                --prefix LD_LIBRARY_PATH : ${
                  pkgs.lib.makeLibraryPath [
                    pkgs.wayland
                    pkgs.libGL
                    pkgs.libglvnd
                    pkgs.libxkbcommon
                    pkgs.onnxruntime
                  ]
                } \
                --set ORT_DYLIB_PATH "${pkgs.onnxruntime}/lib/libonnxruntime.so"
            '';

            meta = with pkgs.lib; {
              description = "GPU-accelerated depth effect wallpaper for Wayland";
              license = licenses.mit;
              platforms = platforms.linux;
              mainProgram = "waydeeper";
            };
          }
        );
      in
      {
        packages = {
          default = waydeeper;
          waydeeper = waydeeper;
        };

        devShells.default = craneLib.devShell {
          inputsFrom = [ waydeeper ];

          packages = with pkgs; [
            rustToolchain
            pkg-config
            cmake
            wayland
            wayland-protocols
            wayland-scanner
            libGL
            libglvnd
            libxkbcommon
            openssl
            onnxruntime
            # Dev tools
            rust-analyzer
            clippy
            rustfmt
          ];

          env = {
            WAYLAND_SCANNER = "${pkgs.wayland}/bin/wayland-scanner";
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
            ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.wayland
              pkgs.libGL
              pkgs.libglvnd
              pkgs.libxkbcommon
              pkgs.openssl
              pkgs.onnxruntime
            ];

          };

          shellHook = ''
            echo "waydeeper-rust development environment"
            echo "  rustc: $(rustc --version)"
            echo "  cargo: $(cargo --version)"
          '';
        };
      }
    )
    // {
      homeManagerModules.default =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.waydeeper;
        in
        {
          options.services.waydeeper = {
            enable = lib.mkEnableOption "Waydeeper depth effect wallpaper daemon";

            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.system}.default;
              description = "The waydeeper package to use.";
            };
          };

          config = lib.mkIf cfg.enable {
            home.packages = [ cfg.package ];

            systemd.user.services.waydeeper = {
              Unit = {
                Description = "Waydeeper depth effect wallpaper daemon";
                After = [ "graphical-session.target" ];
                PartOf = [ "graphical-session.target" ];
              };

              Service = {
                Type = "forking";
                ExecStart = "${lib.getExe cfg.package} daemon";
                ExecStop = "${lib.getExe cfg.package} stop";
                Restart = "on-failure";
              };

              Install = {
                WantedBy = [ "graphical-session.target" ];
              };
            };
          };
        };
    };
}
