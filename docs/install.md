# Installation

QoraPhase requires Python 3.12 or later. Install the released package in a new uv project:

<!-- test -->
```sh
uv init --no-workspace --python 3.12 qoraphase-install-demo
uv add --directory qoraphase-install-demo qoraphase
uv run -q --directory qoraphase-install-demo python -c "import qoraphase"
```

For an existing project, use `uv add qoraphase`; `pip install qoraphase` is another option.
The package includes JAX, NumPy and plotting support. GPU availability depends on your JAX
installation and platform: the [hardware report](hardware.md) records the tested environment
and untested alternatives, not a promise of GPU support on every machine.

## Development and unreleased features

From a clone of this repository, use `uv sync --python 3.12` and `uv run --python 3.12 pytest`.
The `examples` dependency group additionally includes PyVista, scikit-image and SciPy;
install it with `uv sync --group examples` only when an example needs it.
Low-pass progression and masked-pixel diagnostic corrections are on main after the PyPI 0.2.0
release. For those, use a source checkout or install `git+https://github.com/NaNIsaac/QoraPhase`.
Check the [release history](https://github.com/NaNIsaac/QoraPhase/releases) for later published versions.

