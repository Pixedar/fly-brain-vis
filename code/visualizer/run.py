"""
Entry point for the 3D neuron visualizer.

Can be used standalone or called from main.py after simulation.

Usage:
    # Standalone (visualize existing spike data)
    python code/visualizer/run.py --spikes data/results/spikes_sugar.parquet

    # With subset (only sugar GRN neurons and their downstream)
    python code/visualizer/run.py --spikes data/results/spikes_sugar.parquet --subset sugar

    # Custom token
    python code/visualizer/run.py --spikes data/results/spikes_sugar.parquet --token YOUR_TOKEN
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Resolve paths
_CODE_DIR = Path(__file__).resolve().parent.parent
_ROOT_DIR = _CODE_DIR.parent
_DATA_DIR = _ROOT_DIR / 'data'

sys.path.insert(0, str(_CODE_DIR))


DEFAULT_TOKEN = "313ed195cd87a8456f806c11daba2435"


def _load_id_mapping(comp_path=None):
    """Load neuron ID mapping from completeness CSV.

    Returns:
        (flyid2i, i2flyid, all_root_ids)
    """
    comp_path = comp_path or _DATA_DIR / '2025_Completeness_783.csv'
    df = pd.read_csv(comp_path, index_col=0)
    all_root_ids = list(df.index)
    flyid2i = {rid: i for i, rid in enumerate(all_root_ids)}
    i2flyid = {i: rid for rid, i in flyid2i.items()}
    return flyid2i, i2flyid, all_root_ids


def _get_subset_ids(subset_name, spike_path=None):
    """Get root IDs for a named subset or from spike data.

    Subsets:
    - 'sugar': 21 sugar GRN neurons + all neurons that spiked
    - 'p9': P9 neurons + all neurons that spiked
    - 'active': all neurons that spiked in the simulation
    - 'active_N': top N most active neurons
    """
    from benchmark import EXPERIMENTS

    # Start with experiment neurons if applicable
    subset = set()
    if subset_name in EXPERIMENTS:
        exp = EXPERIMENTS[subset_name]
        subset.update(exp['neu_exc'])
        subset.update(exp.get('neu_exc2', []))

    # Add all spiking neurons from the data
    if spike_path and Path(spike_path).exists():
        df = pd.read_parquet(spike_path)
        spiking_ids = set(df['flywire_id'].unique())

        if subset_name.startswith('active_'):
            # Top N most active
            try:
                n = int(subset_name.split('_')[1])
            except (ValueError, IndexError):
                n = 1000
            counts = df.groupby('flywire_id').size().nlargest(n)
            subset.update(counts.index.tolist())
        elif subset_name == 'active':
            subset.update(spiking_ids)
        else:
            # Named experiment: add downstream spiking neurons
            subset.update(spiking_ids)

    return subset if subset else None


def _load_connectivity(subset_ids=None):
    """Load connectivity data with weights and excitatory/inhibitory info.

    Uses vectorized numpy operations for speed (15M rows).

    Args:
        subset_ids: optional set of root IDs to filter to

    Returns:
        (pre_ids, post_ids, weights, excitatory) or None
    """
    conn_path = _DATA_DIR / '2025_Connectivity_783.parquet'
    if not conn_path.exists():
        return None

    _, i2flyid, _ = _load_id_mapping()

    # Build vectorized index->flyid lookup array
    max_idx = max(i2flyid.keys()) + 1
    idx_to_rid = np.zeros(max_idx, dtype=np.int64)
    for i, rid in i2flyid.items():
        idx_to_rid[i] = rid

    print("[visualizer]   Loading connectivity parquet...")
    df = pd.read_parquet(conn_path)
    pre_indices = df['Presynaptic_Index'].values
    post_indices = df['Postsynaptic_Index'].values
    weights = df['Connectivity'].values.astype(np.float32)
    excitatory = df['Excitatory'].values.astype(np.int8)

    # Vectorized ID mapping
    valid_pre = pre_indices < max_idx
    valid_post = post_indices < max_idx
    valid = valid_pre & valid_post

    pre_ids = np.zeros(len(pre_indices), dtype=np.int64)
    post_ids = np.zeros(len(post_indices), dtype=np.int64)
    pre_ids[valid] = idx_to_rid[pre_indices[valid]]
    post_ids[valid] = idx_to_rid[post_indices[valid]]

    if subset_ids:
        # Vectorized subset filtering using a set-based approach
        subset_arr = np.array(sorted(subset_ids), dtype=np.int64)
        pre_in = np.isin(pre_ids, subset_arr)
        post_in = np.isin(post_ids, subset_arr)
        mask = pre_in & post_in
        pre_ids = pre_ids[mask]
        post_ids = post_ids[mask]
        weights = weights[mask]
        excitatory = excitatory[mask]

    return pre_ids, post_ids, weights, excitatory


def launch_visualizer(spike_path, experiment=None, subset=None,
                      token=None, connections=False, skeletons=False):
    """Launch the 3D brain visualizer.

    Args:
        spike_path: path to spike parquet file from simulation
        experiment: experiment config dict (optional, for subset naming)
        subset: subset mode - 'sugar', 'p9', 'active', 'active_N', or None for all
        token: FlyWire API token
        connections: whether to render synapse connections
        skeletons: whether to fetch and render neuron skeletons (subset mode only)
    """
    try:
        from .fetch_geometry import fetch_neuron_positions, fetch_skeletons, setup_flywire_token
        from .spike_player import SpikePlayer
        from .renderer import BrainRenderer
    except ImportError:
        from fetch_geometry import fetch_neuron_positions, fetch_skeletons, setup_flywire_token
        from spike_player import SpikePlayer
        from renderer import BrainRenderer

    token = token or DEFAULT_TOKEN

    # Load neuron ID mappings
    print("[visualizer] Loading neuron ID mappings...")
    flyid2i, i2flyid, all_root_ids = _load_id_mapping()

    # Determine which neurons to visualize
    subset_ids = None
    if subset:
        subset_ids = _get_subset_ids(subset, spike_path)
        if subset_ids:
            print(f"[visualizer] Subset '{subset}': {len(subset_ids)} neurons")

    active_ids = sorted(subset_ids) if subset_ids else all_root_ids

    # Fetch 3D positions from FlyWire
    setup_flywire_token(token)
    positions = fetch_neuron_positions(active_ids, token=token)

    # Optionally fetch skeletons for subset mode
    skel_data = None
    if skeletons and subset_ids and len(subset_ids) <= 500:
        print(f"[visualizer] Fetching skeletons for {len(subset_ids)} neurons...")
        skel_data = fetch_skeletons(list(subset_ids), token=token)

    # Load connectivity with weights
    conn_data = None
    conn_weights = None
    conn_excitatory = None
    print("[visualizer] Loading connectivity...")
    result = _load_connectivity(subset_ids)
    if result is not None:
        pre_ids, post_ids, conn_weights, conn_excitatory = result
        conn_data = (pre_ids, post_ids)
        print(f"[visualizer] Loaded {len(pre_ids)} connections "
              f"(weights range: {conn_weights.min():.0f}-{conn_weights.max():.0f})")

    # Create spike player - slow playback so spikes are visible
    player = SpikePlayer(
        spike_path=spike_path,
        neuron_index=flyid2i,  # Use full mapping, renderer will remap
        playback_speed=0.1,    # Slow: 100ms simulation plays over ~10 seconds
        decay_ms=80.0,         # Longer glow trail for dramatic effect
    )

    # Create renderer
    renderer = BrainRenderer(
        positions=positions,
        connectivity=conn_data,
        conn_weights=conn_weights,
        conn_excitatory=conn_excitatory,
        skeletons=skel_data,
        subset_ids=subset_ids,
        neuron_ids=active_ids,
    )

    # Launch
    print(f"[visualizer] Launching 3D viewer with {len(active_ids)} neurons...")
    renderer.start(player)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='3D Drosophila Brain Neural Activity Visualizer'
    )
    parser.add_argument('--spikes', type=str, required=True,
                        help='Path to spike parquet file from simulation')
    parser.add_argument('--subset', type=str, default=None,
                        help='Neuron subset: "sugar", "p9", "active", "active_N" '
                             '(N = top N neurons), or omit for full brain')
    parser.add_argument('--token', type=str, default=DEFAULT_TOKEN,
                        help='FlyWire API token')
    parser.add_argument('--connections', action='store_true',
                        help='Render synapse connection lines')
    parser.add_argument('--skeletons', action='store_true',
                        help='Fetch and render neuron skeletons (subset mode, max 500)')
    parser.add_argument('--speed', type=float, default=0.5,
                        help='Initial playback speed (default: 0.5)')

    args = parser.parse_args()

    spike_path = Path(args.spikes)
    if not spike_path.exists():
        # Try finding the most recent spike file
        results_dir = _DATA_DIR / 'results'
        parquets = sorted(results_dir.glob('*.parquet'), key=lambda p: p.stat().st_mtime)
        if parquets:
            spike_path = parquets[-1]
            print(f"[visualizer] Using most recent spike file: {spike_path}")
        else:
            print(f"Error: No spike file found at {args.spikes}")
            print("Run a simulation first: python main.py --brian2-cpu --t_run 0.1 --n_run 1")
            sys.exit(1)

    launch_visualizer(
        spike_path=str(spike_path),
        subset=args.subset,
        token=args.token,
        connections=args.connections,
        skeletons=args.skeletons,
    )


if __name__ == '__main__':
    main()
