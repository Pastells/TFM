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
