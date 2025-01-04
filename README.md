# Installation
This project requires GDAL, which can be a pain to install. Pip is unable to install it, so there are two recommnended ways:

1. Install it system wide. e.g. `sudo apt-get install gdal-bin`
1. Use conda. e.g. `conda install gdal` (after creating a conda environment for this project)

Otherwise, the rest of the dependencies can be installed via pip. The following steps uses the pip-based environment/dependency manager `uv`. Install uv with:

```bash
pip install uv
```

Clone the project and install the dependencies with:

```bash
git clone git@github.com:alexmitchell/map_dataset.git
cd map_dataset
uv sync
```

# Running
If you used `uv` to install the project, you can run scripts from the command line with:

```bash
uv run <script> <args>
```

Or can run scripts in interactive mode in your IDE like VSCode.