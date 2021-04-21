Modelling for science and engineering Master thesis

Requirements specified in requirements.txt (python) and manifest (julia).

Manual installation:

- Python: numpy, scipy, pandas, julia\
  `pip install numpy scipy pandas julia`

- Julia: with PyCall, OrdinaryDiffEq, StaticArrays\
  `julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.add("OrdinaryDiffEq"); Pkg.add("StaticArrays")'`

- OrdinaryDiffEq, StaticArrays are found precompiled in sysimage/ODEs.so

Model executed like: `python src/self_force.py data/prova.csv`
