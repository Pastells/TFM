using Pkg

# Create virtual environment
Pkg.activate(".")

# Install packages
Pkg.add("PackageCompiler")
Pkg.add("PyCall")
Pkg.add("OrdinaryDiffEq")
Pkg.add("StaticArrays")
Pkg.add("Markdown")
Pkg.add("Plots")

# Create sysimage
using PackageCompiler
create_sysimage([:OrdinaryDiffEq, :StaticArrays, :Markdown, :Plots],
    sysimage_path="sysimage.so",
    precompile_execution_file=["src/mode_comp.jl","src/compactification.jl"])
