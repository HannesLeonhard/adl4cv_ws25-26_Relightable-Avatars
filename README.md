# Relightable Avatars from Material-Aware Diffusion Models

## Setup Environments

Initialize the submodules `rgbx` and `diffusion_renderer`.
```
git submodule update --init --recursive
```

### Setup rgbx

1. Setup conda environment according to the instructions in `rgbx/`
2. Install current project so that it is available in notebooks with `pip install -e .`
3. For code quality install `conda install ruff`. Code can the be formatted with `ruff format .`

### Setup diffusion renderer

1. Setup conda environment according to the instructions in `diffusion_renderer/`
2. Install current project so that it is availabel in notebooks with `pip install -e .`
3. For code quality install `conda install ruff`. Code can the be formatted with `ruff format .` 
