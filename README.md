# fly-brain-vis: 3D Neural Activity Visualizer for *Drosophila*

![Demo](assets/demo.gif)

> **This is a visualization companion project based on
> [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain).**
> It is **not** the official upstream repository.
> The core LIF simulation engine, connectome data, and benchmark
> infrastructure all originate from the upstream project.

Real-time 3D visualization of whole-brain spike activity in the adult
fruit fly, rendered with GPU-accelerated gaussian-splat glow effects on
top of the [FlyWire](https://flywire.ai/) connectome (~138 k neurons,
~5 M synapses).

## What this adds

| Feature | Description |
|---|---|
| **Gaussian-splat neuron glow** | Multi-layer VTK PointGaussianMapper with custom GLSL shaders (aura / bloom / core / hot-white) |
| **Animated connections** | Synaptic lines flash in sync with presynaptic spikes, modulated by connection weight |
| **FlyWire anatomical coordinates** | Neuron positions fetched from FlyWire L2 cache and cached locally |
| **Free camera** | Rotate, zoom, and pan with mouse; keyboard speed/pause controls |
| **`--visualize` flag** | One command: `python main.py --pytorch --t_run 0.1 --n_run 1 --visualize` |
| **`--export-activity`** | Machine-readable spike output path for external tool consumption |

## FlyWire API Token

A FlyWire chunkedgraph token is required to fetch neuron positions.
Get one at: **https://global.daf-apis.com/auth/api/v1/create_token**

Then set it as an environment variable:
```bash
# Linux / macOS
export FLYWIRE_TOKEN=your_token_here
# Windows (PowerShell)
$env:FLYWIRE_TOKEN = "your_token_here"
```

Or pass it directly via `--token`:
```bash
python code/visualizer/run.py --spikes data/results/pytorch_t0.1s_n1.parquet --token your_token_here
```

> **Note:** After the first run, neuron positions are cached locally in
> `data/cache/`. Subsequent runs don't need network access.

## Quick start

```bash
# 1. Install dependencies (adds fury, fafbseg, navis on top of upstream env)
conda env create -f environment.yml
conda activate brain-fly

# 2. Run simulation + launch visualizer
python main.py --pytorch --t_run 0.1 --n_run 1 --no_log_file --visualize

# 3. Or visualize existing spike data standalone
python code/visualizer/run.py --spikes data/results/pytorch_t0.1s_n1.parquet
```

### Visualizer controls

| Input | Action |
|---|---|
| Space | Pause / Resume |
| R | Reset playback |
| +/- | Speed up / down |
| C | Toggle connections on/off |
| D | Toggle connection diff mode (highlights sudden activity changes) |
| Click | Select neuron — shows FlyWire ID, in/out degree, activity |
| Click again | Highlight pre/post-synaptic partners |

## Upstream integration

This fork includes a minimal `proposal/export-activity` branch that adds
only a `--export-activity` flag and documents the spike output schema.
That branch is intended as a clean, low-friction pull request to the
upstream repository -- it changes nothing about the simulation, only adds
discoverability for the existing spike export so external tools can
consume it.

## Architecture

```
fly-brain-vis/
├── main.py                     # CLI: simulation + optional --visualize
├── code/
│   ├── benchmark.py            # Simulation orchestrator (from upstream)
│   ├── run_pytorch.py          # PyTorch LIF backend (from upstream)
│   ├── run_brian2_cuda.py      # Brian2/Brian2CUDA backend (from upstream)
│   ├── run_nestgpu.py          # NEST GPU backend (from upstream)
│   └── visualizer/             # << new: 3D visualization module
│       ├── __init__.py
│       ├── run.py              # Entry point / CLI
│       ├── renderer.py         # FURY/VTK gaussian-splat renderer
│       ├── spike_player.py     # Spike data playback engine
│       └── fetch_geometry.py   # FlyWire position/skeleton fetcher + cache
├── data/
│   ├── 2025_Completeness_783.csv
│   ├── 2025_Connectivity_783.parquet
│   └── cache/                  # Auto-generated position/skeleton caches
└── environment.yml             # Conda env (upstream + visualization deps)
```

## Original simulation documentation

The core simulation engine is documented in the upstream repository:
**[eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)**

All four simulation backends (Brian2, Brian2CUDA, PyTorch, NEST GPU),
the connectome data, experiment definitions, and benchmark infrastructure
are maintained upstream. See the upstream README for full simulation
documentation, installation of NEST GPU, system requirements, and
benchmark results.

### Simulation quick reference

```bash
# Run all backends with default settings
python main.py

# Single backend, specific duration
python main.py --pytorch --t_run 1 --n_run 1

# Export spike data for external tools
python main.py --pytorch --t_run 0.1 --n_run 1 --export-activity
```

Spike output format (parquet): columns `t` (ms), `trial`, `flywire_id`, `exp_name`.

## License

This project inherits its license from the upstream repository.
See [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
for details.
