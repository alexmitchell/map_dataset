# Installation
## GDAL
This project requires GDAL, which can be a pain to install. Pip is unable to
install it, so there are two recommnended ways:

1. Install it system wide. e.g. `sudo apt-get install gdal-bin`
1. Use conda. e.g. `conda install gdal` (after creating a conda environment for
this project)

Both methods seem to work just about as well. If using UV later, Conda will have
some redundancy with python environments. (Conda has a python for the
environment, then UV has another python for the project venv. Kind of icky but
seems to work okay)

## With UV
Otherwise, the rest of the dependencies can be installed via pip. The following
steps uses the pip-based environment/dependency manager `uv`. Install uv with:

```bash
pip install uv
```

Clone the project:
```bash
git clone git@github.com:alexmitchell/map_dataset.git
cd map_dataset
uv sync
```

The `pyproject.toml` was created following instructions on
[UV's guide](https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies)
to facilitate installing pytorch with CPU or CUDA GPU processing. If you need a
CUDA version not listed below, follow the guide to add it to the
`pyproject.toml`.


To install all dependencies with the CPU version of Pytorch, run:
```bash
uv sync --extra cpu
```

To install all dependencies with the CUDA version of Pytorch, run (NOTE: I don't
have a GPU so can't test the CUDA install):
```bash
uv sync --extra cu124
```

## Without UV
Dependencies are listed in `pyproject.toml`. Good luck :P

For pytorch, follow the instructions on the
[Pytorch website](https://pytorch.org/get-started/locally/).


# Running
If you used `uv` to install the project, you can run scripts from the command
line with:

```bash
uv run <script> <args>
```

Or can run scripts in interactive mode in your IDE like VSCode. (If it
recognizes the `# %%` syntax for cells)