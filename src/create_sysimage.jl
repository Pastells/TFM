using PackageCompiler
using Pkg
Pkg.activate(".")
create_sysimage([:StaticArrays, :Markdown],
                sysimage_path="sysimage2.so")
