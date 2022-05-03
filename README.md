Modelling for science and engineering Master thesis

Requirements specified in requirements.txt (python) and manifest (julia).

Installation instructions:

- Python:

  `pip install numpy scipy pandas julia` \

  Or in a conda virtual environment:

  ```
  conda create --yes -n venv pip numpy pandas scipy
  conda activate venv
  ```

- Julia:
  - download from https://julialang.org/downloads/
  - Install needed packages and precompile a sysimage with:
    `julia src/create_sysimage.jl`
  - If no sysimage is wanted, remove `sysimage="sysimage.so"` from the Julia
    call in self_force.py.
  - Install py-julia, to call Julia from Python: `python3 -m pip install julia`

Model executed like:
`python src/self_force.py data/prova.csv [-log_print] [-save]`

Read results into pandas:
`df = pd.read_csv("data/prova_results.csv", comment="#", names=["rp","fm","fp"])`

### Abstract

The self-force problem arises in the description of the motion of particles under the action of
physical fields. It has to do with the singularities that emerge when we estimate the action of
the field created by a particle on its own motion. In this work we present a new method for
the computation of the self-force acting on a particle moving under the influence of a scalar
field in the spacetime geometry of a non-rotating (Schwarzschild) Black Hole.
