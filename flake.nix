{
  description = "Waydeeper - GPU-accelerated depth effect wallpaper for Wayland";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        waydeeper = pkgs.python3Packages.buildPythonApplication {
          pname = "waydeeper";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = [
            pkgs.gobject-introspection
            pkgs.wrapGAppsHook4
            pkgs.makeWrapper
          ]
          ++ (with pkgs.python3Packages; [
            setuptools
            wheel
          ]);

          buildInputs = [
            pkgs.gtk4
            pkgs.gtk4-layer-shell
            pkgs.libadwaita
            pkgs.wayland
            pkgs.libGL
            pkgs.libepoxy
            pkgs.glib
          ];

          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            pillow
            onnxruntime
            pygobject3
            pyopengl
            pyopengl-accelerate
          ];

          makeWrapperArgs = [
            "--prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [ pkgs.gtk4-layer-shell ]}"
          ];

          meta = with pkgs.lib; {
            description = "GPU-accelerated depth effect wallpaper for Wayland";
            license = licenses.mit;
            platforms = platforms.linux;
            mainProgram = "waydeeper";
          };
        };
      in
      {
        packages = {
          default = waydeeper;
          waydeeper = waydeeper;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (python3.withPackages (
              ps: with ps; [
                numpy
                pillow
                onnxruntime
                pygobject3
                pyopengl
                pyopengl-accelerate
                setuptools
                black
                pytest
                mypy
              ]
            ))
            pkg-config
            gobject-introspection
            gtk4
            gtk4-layer-shell
            libadwaita
            wayland
            wayland-protocols
            libGL
            libepoxy
            glib
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
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
