{
  description = "Scratch dir for various testing";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixpkgs-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    pkgsArm = nixpkgs.legacyPackages."aarch64-linux";
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        jupyter
    
          (python312.withPackages (ps: with ps; [
      matplotlib
      numpy
      pip
      jupyter
      pandas
      scikit-learn
      notebook
      ipympl
    ]))
        ];
    };
  };
}

#{
#  description = "Scratch dir for various testing";
#
##  outputs = {nixpkgs, ...}: let
##    system = "x86_64-linux";
##    pkgs = nixpkgs.legacyPackages.${system};
##  in {
##    devShells.${system}.default = pkgs.mkShell {
##      packages = with pkgs; [
##        gcc8
##        python310	
##        gdb
##        man-pages
##
##        linuxHeaders
##      ];
##    };
##  };
#
#
#
#  outputs = { self, nixpkgs }: let
#   system = "x86_64-linux";
#   pkgs = nixpkgs.legacyPackages.${system}; in {
#    devShells.${system}.default = pkgs.lib.mkShell {
#      buildInputs = with pkgs; [
#      ];
#
#      # FHS User environment
#      shellHook = ''
#        echo "Entering FHS environment"
#      '';
#
#      fhs = nixpkgs.buildFHSUserEnv {
#        name = "fhs-devshell";
#        targetPkgs = with pkgs; [
#gcc8
#python310
#gdb
#man-pages
#linuxHeaders
#coreutils-full
#        ];
#        runScript = "bash";
#      };
#    };
#  };
#}


# flake.nix
#{
#  inputs = {
#    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
#  };
#
#  outputs = { self, nixpkgs }:
#    let
#      system = "x86_64-linux";
#      pkgs = nixpkgs.legacyPackages.${system};
#      fhs = pkgs.buildFHSEnv {
#        name = "fhs-shell";
#        targetPkgs = pkgs: with pkgs; [
#    
#
#
#        
#        ] ;
#      };
#    in
#      {
#        devShells.${system}.default = fhs.env;
#      };
#}
