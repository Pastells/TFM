Modelling for science and engineering Master thesis

Requirements specified in requirements.txt (python) and manifest (julia).

Manual installation:

- Python: numpy, scipy, pandas, julia\
  `pip install numpy scipy pandas julia`

- Julia: with PyCall, OrdinaryDiffEq, StaticArrays, Markdown\
  `julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.add("OrdinaryDiffEq"); Pkg.add("StaticArrays"); Pkg.add("Markdown")'`

- OrdinaryDiffEq, StaticArrays can be precompiled to sysimage.so\
   To create it, run: `julia src/create_sysimage.jl` Then uncomment the line in
  self_force.py containing `sysimage` and comment the line above.

Model executed like:
`python src/self_force.py data/prova.csv [-log_print] [-save]`
