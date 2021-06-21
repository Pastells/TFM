using PackageCompiler
using Pkg
Pkg.activate(".")
# Pkg.add("Plots")
create_sysimage([:OrdinaryDiffEq, :StaticArrays, :Markdown, :Plots],
    sysimage_path="sysimage_plots.so",
    precompile_execution_file=["src/mode_comp.jl","src/compactification.jl"])
