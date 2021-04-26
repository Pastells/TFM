Modelling for science and engineering Master thesis

Requirements specified in requirements.txt (python) and manifest (julia).

Manual installation:

- Python: numpy, scipy, pandas, julia\
  `pip install numpy scipy pandas julia`

- Julia: with PyCall, OrdinaryDiffEq, StaticArrays, Markdown\
  `julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.add("OrdinaryDiffEq"); Pkg.add("StaticArrays")'`

- OrdinaryDiffEq, StaticArrays are found precompiled in sysimage/ODEs.so\
  The sysimage may be dependent on the julia version and other stuff. If it fails
  to execute, comment the line in self_force.py containing `sysimage` and uncomment
  the line below.

Model executed like: `python src/self_force.py data/prova.csv`
