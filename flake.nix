{
  description = "IronKaggleProject with Python environment managed by uv";

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

        # Main packages for development including uv, jupyter, and zsh
        mainPackages = with pkgs; [
          uv
          jupyter
          zsh
          marp-cli
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = mainPackages;

          # Shell hook to setup Python environment with uv
          shellHook = ''
            echo "Entering development environment with uv-managed Python packages"
            echo "Run 'uv sync' to install dependencies from pyproject.toml"
            echo "Run 'uv run jupyter notebook' to start Jupyter"
            export SHELL=${pkgs.zsh}/bin/zsh
            exec zsh
          '';
        };

        packages = {
          # Package that can be built with uv-managed dependencies
          default = pkgs.stdenv.mkDerivation {
            name = "IronKaggleProject-uv-env";
            src = ./.;

            buildInputs = mainPackages;

            installPhase = "true";
            buildPhase = "true";
            fixupPhase = "true";

            dontConfigure = true;
            doInstall = true;
          };
        };
      }
    );
}
