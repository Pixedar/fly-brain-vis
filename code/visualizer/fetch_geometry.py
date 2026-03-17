"""
Fetch and cache neuron 3D positions and skeletons from FlyWire.

Uses fafbseg L2 cache for bulk centroid lookups (works with chunkedgraph token).
Falls back to CAVE materialization if available.
All results are cached locally to avoid repeated network requests.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# FlyWire voxel resolution: 4x4x40 nm
VOXEL_RESOLUTION = np.array([4.0, 4.0, 40.0])

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'cache'
POSITIONS_CACHE = CACHE_DIR / 'neuron_positions_783.parquet'
SKELETONS_CACHE = CACHE_DIR / 'skeletons'


def setup_flywire_token(token):
    """Configure FlyWire API token for fafbseg."""
    os.environ['CHUNKEDGRAPH_SECRET'] = token
    os.environ['CAVE_SECRET'] = token
    try:
        from fafbseg import flywire
        flywire.set_chunkedgraph_secret(token)
    except Exception:
        pass


def _fetch_positions_via_l2(root_ids):
    """Fetch positions for multiple neurons via fafbseg get_l2_info.

    Uses bounds_nm from the L2 cache to compute centroids.
    Works with just the chunkedgraph token (no CAVE permissions needed).
    """
    from fafbseg import flywire
    positions = {}
    total = len(root_ids)
    BATCH_SIZE = 200

    for i in range(0, total, BATCH_SIZE):
        batch = root_ids[i:i + BATCH_SIZE]
        print(f"[visualizer]   L2 batch {i // BATCH_SIZE + 1}: "
              f"querying {len(batch)} neurons ({i}/{total})...")
        try:
            info = flywire.get_l2_info(batch, progress=False)
            for _, row in info.iterrows():
                bounds = np.array(row['bounds_nm'])
                centroid = (bounds[:3] + bounds[3:]) / 2.0
                positions[int(row['root_id'])] = tuple(centroid)
        except Exception as e:
            print(f"[visualizer]   Warning: L2 batch failed: {e}")
            # Try one at a time for this batch
            for rid in batch:
                try:
                    info = flywire.get_l2_info(rid, progress=False)
                    bounds = np.array(info['bounds_nm'].iloc[0])
                    centroid = (bounds[:3] + bounds[3:]) / 2.0
                    positions[int(rid)] = tuple(centroid)
                except Exception:
                    pass

    return positions


def _fetch_positions_via_cave(root_ids, token=None):
    """Fetch positions via CAVE materialization (requires view permission)."""
    try:
        import caveclient
        client = caveclient.CAVEclient("flywire_fafb_production")
        if token:
            client.auth.token = token

        positions = {}
        BATCH_SIZE = 50000

        for i in range(0, len(root_ids), BATCH_SIZE):
            batch = root_ids[i:i + BATCH_SIZE]
            print(f"[visualizer]   CAVE batch {i // BATCH_SIZE + 1}: "
                  f"querying {len(batch)} neurons...")
            nuc_df = client.materialize.query_table(
                "nuclei_v1",
                filter_in_dict={"pt_root_id": batch},
            )
            for _, row in nuc_df.iterrows():
                rid = int(row['pt_root_id'])
                pos = np.array(row['pt_position']) * VOXEL_RESOLUTION
                positions[rid] = tuple(pos)

        return positions
    except Exception as e:
        print(f"[visualizer]   CAVE query failed: {e}")
        return {}


def fetch_neuron_positions(root_ids, cache_path=None, token=None):
    """Fetch neuron centroid positions from FlyWire.

    Strategy:
    1. Check local cache first
    2. Try CAVE materialization (fast bulk query, needs view permission)
    3. Fall back to L2 cache centroids (slower but works with basic token)
    4. Assign random positions for any remaining neurons

    Args:
        root_ids: array-like of FlyWire root IDs (integers)
        cache_path: path to cache file
        token: FlyWire API token

    Returns:
        dict mapping root_id (int) -> (x, y, z) in nanometers
    """
    cache_path = Path(cache_path) if cache_path else POSITIONS_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache_path.exists():
        print(f"[visualizer] Loading cached positions from {cache_path}")
        df = pd.read_parquet(cache_path)
        positions = {
            int(row.root_id): (row.x, row.y, row.z)
            for row in df.itertuples()
        }
        missing = set(root_ids) - set(positions.keys())
        if not missing:
            return positions
        print(f"[visualizer] Cache has {len(positions)} positions, "
              f"missing {len(missing)}, fetching remainder...")
    else:
        positions = {}
        missing = set(root_ids)

    if token:
        setup_flywire_token(token)

    missing_list = sorted(missing)
    print(f"[visualizer] Fetching {len(missing_list)} neuron positions from FlyWire...")
    print("[visualizer] This may take a few minutes on first run (results are cached).")

    # Try CAVE first (fast)
    print("[visualizer] Trying CAVE materialization...")
    new_positions = _fetch_positions_via_cave(missing_list, token)

    if len(new_positions) < len(missing_list):
        still_missing = [rid for rid in missing_list if rid not in new_positions]
        print(f"[visualizer] CAVE returned {len(new_positions)} positions, "
              f"fetching {len(still_missing)} via L2 cache...")
        l2_positions = _fetch_positions_via_l2(still_missing)
        new_positions.update(l2_positions)

    positions.update(new_positions)

    # For neurons we couldn't locate, assign random positions in brain bbox
    still_missing = set(root_ids) - set(positions.keys())
    if still_missing:
        if positions:
            print(f"[visualizer] Warning: {len(still_missing)} neurons have no position data, "
                  "using interpolated positions")
            known_pos = np.array(list(positions.values()))
            bbox_min = known_pos.min(axis=0)
            bbox_max = known_pos.max(axis=0)
            rng = np.random.default_rng(42)
            for rid in still_missing:
                pos = rng.uniform(bbox_min, bbox_max)
                positions[int(rid)] = tuple(pos)
        else:
            # No positions at all — generate a sphere layout as placeholder
            print("[visualizer] Warning: No positions fetched. Using spherical layout.")
            n = len(still_missing)
            rng = np.random.default_rng(42)
            phi = rng.uniform(0, 2 * np.pi, n)
            costheta = rng.uniform(-1, 1, n)
            theta = np.arccos(costheta)
            r = 100.0 * rng.uniform(0.3, 1.0, n) ** (1.0 / 3.0)
            for i, rid in enumerate(sorted(still_missing)):
                x = r[i] * np.sin(theta[i]) * np.cos(phi[i])
                y = r[i] * np.sin(theta[i]) * np.sin(phi[i])
                z = r[i] * np.cos(theta[i])
                positions[int(rid)] = (x, y, z)

    # Save to cache
    if positions:
        df = pd.DataFrame([
            {'root_id': rid, 'x': p[0], 'y': p[1], 'z': p[2]}
            for rid, p in positions.items()
        ])
        df.to_parquet(cache_path, index=False)
        print(f"[visualizer] Cached {len(positions)} positions to {cache_path}")

    return positions


def fetch_skeletons(root_ids, cache_dir=None, token=None):
    """Fetch neuron skeletons from FlyWire for detailed subset rendering.

    Args:
        root_ids: list of FlyWire root IDs
        cache_dir: directory for skeleton cache files
        token: FlyWire API token

    Returns:
        dict mapping root_id -> dict with 'nodes' (Nx3 array) and 'edges' (Mx2 array)
    """
    cache_dir = Path(cache_dir) if cache_dir else SKELETONS_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    if token:
        setup_flywire_token(token)

    skeletons = {}

    for rid in root_ids:
        cache_file = cache_dir / f"{rid}.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            skeletons[int(rid)] = {
                'nodes': data['nodes'],
                'edges': data['edges'],
            }
            continue

        print(f"[visualizer] Fetching skeleton for {rid}...")
        try:
            from fafbseg import flywire
            import navis

            sk = flywire.get_skeletons(rid, progress=False)
            if isinstance(sk, navis.NeuronList):
                sk = sk[0]

            nodes = sk.nodes[['x', 'y', 'z']].values.astype(np.float64)
            node_ids = sk.nodes['node_id'].values
            parent_ids = sk.nodes['parent_id'].values
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            edges = []
            for i, pid in enumerate(parent_ids):
                if pid >= 0 and pid in id_to_idx:
                    edges.append([i, id_to_idx[pid]])
            edges = np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)

            skeletons[int(rid)] = {'nodes': nodes, 'edges': edges}
            np.savez_compressed(cache_file, nodes=nodes, edges=edges)

        except Exception as e:
            print(f"[visualizer]   Warning: failed to fetch skeleton for {rid}: {e}")

    return skeletons


def fetch_synapse_positions(pre_ids, post_ids, positions):
    """Get synapse line endpoints from pre/post neuron positions.

    Uses neuron centroid positions as endpoints for connection lines.
    """
    starts = []
    ends = []
    for pre, post in zip(pre_ids, post_ids):
        if pre in positions and post in positions:
            starts.append(positions[pre])
            ends.append(positions[post])
    return np.array(starts, dtype=np.float64), np.array(ends, dtype=np.float64)
