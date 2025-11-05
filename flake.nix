{
  description = "IronKaggleProject with Python environment managed by uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Define the main Python packages that can be reused for building artifacts
        mainPythonPackages = ps: with ps; [
          ps.numpy
          ps.pandas
          ps.matplotlib
          ps.seaborn
          ps.plotly
          ps.scikit-learn
          ps.ipython
        ];
        
        # Python environment with the main packages
        # pythonEnv = pkgs.python311.withPackages mainPythonPackages;
        
        # Main packages including uv and jupyter for development
        mainPackages = with pkgs; [
          # pythonEnv
          uv
          jupyter
          zsh 
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = mainPackages;
          
          shellHook = ''
            zsh
          '';
        };
        
        packages = {
          # Main application package that can be built/reused
          default = pkgs.python311Packages.buildPythonApplication {
            pname = "IronKaggleProject";
            version = "0.1.0";
            src = ./.;
            pyproject = true;
            build-system = [ pkgs.python311Packages.setuptools ];
            propagatedBuildInputs = mainPythonPackages pkgs.python311Packages;
            doCheck = false; # Skip tests for now
          };
          
          # Python environment as a package for reuse
          # pythonEnv = pythonEnv;
        };
      });
}
