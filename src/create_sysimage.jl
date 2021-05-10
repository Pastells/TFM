using PackageCompiler
using Pkg
Pkg.activate(".")
create_sysimage([:OrdinaryDiffEq, :StaticArrays, :Markdown],
    sysimage_path="sysimage.so",
    precompile_execution_file=["src/mode_comp.jl","src/hyperboloidal_compactification_tanh.jl"])
