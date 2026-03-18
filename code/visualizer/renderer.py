"""
FURY/VTK GPU-accelerated 3D brain renderer with gaussian splat glow.

Renders neurons using layered gaussian splat actors (aura/bloom/core/hot)
for beautiful fire-like neuron flashes, plus animated connection lines
that propagate signals modulated by synaptic weight.

Visual design adapted from fury_neuron_flash_playground_v10.py:
  - Multi-layer gaussian splats with custom GLSL shaders
  - Warm ember color ramp: black -> deep red -> orange -> amber -> yellow -> white
  - Base points suppressed inside hot regions
  - Additive blending for glow/bloom

All neuron firing is driven by real simulation spike data.
All positions and connections are from real FlyWire anatomy.

Performance mode (--performance):
  - Uses 2 splat layers instead of 4 (core + hot only)
  - Skips per-frame base point suppression (static base points)
  - Limits connections to 15k and skips connection updates every other frame
  - Good for 100k+ neuron rendering on mid-range GPUs
"""

import numpy as np
from fury import window, ui
try:
    import vtk
    from vtk.util import numpy_support
except Exception as e:
    raise RuntimeError("VTK is required (normally installed with FURY).") from e


# ============================================================================
# Color Constants (from playground v10 warm fire palette)
# ============================================================================
BLACK = (0.0, 0.0, 0.0)
DEEP_RED = (0.08, 0.00, 0.00)
BLOOD_RED = (0.30, 0.01, 0.00)
ORANGE_RED = (0.92, 0.07, 0.01)
AMBER = (1.00, 0.43, 0.02)
YELLOW = (1.00, 0.84, 0.12)
PALE_YELLOW = (1.00, 0.93, 0.62)
WARM_WHITE = (1.00, 0.992, 0.972)

# Excitatory connection color ramp: blue → violet → white
DEEP_INDIGO = (0.02, 0.00, 0.08)
DARK_BLUE = (0.04, 0.02, 0.28)
ELECTRIC_BLUE = (0.10, 0.18, 0.82)
BLUE_VIOLET = (0.38, 0.22, 0.92)
VIOLET = (0.62, 0.36, 1.00)
LIGHT_VIOLET = (0.82, 0.68, 1.00)
CONN_WHITE = (0.94, 0.92, 1.00)

CONN_EXC_STOPS = [
    (0.00, BLACK),
    (0.04, DEEP_INDIGO),
    (0.16, DARK_BLUE),
    (0.42, ELECTRIC_BLUE),
    (0.68, BLUE_VIOLET),
    (0.84, VIOLET),
    (0.94, LIGHT_VIOLET),
    (1.00, CONN_WHITE),
]

# Inhibitory connection color ramp: dark teal → cyan → white
DEEP_TEAL = (0.00, 0.04, 0.06)
DARK_CYAN = (0.01, 0.12, 0.22)
TEAL = (0.02, 0.38, 0.48)
CYAN = (0.08, 0.72, 0.78)
LIGHT_CYAN = (0.42, 0.92, 0.88)
CYAN_WHITE = (0.88, 1.00, 0.98)

CONN_INH_STOPS = [
    (0.00, BLACK),
    (0.04, DEEP_TEAL),
    (0.16, DARK_CYAN),
    (0.42, TEAL),
    (0.68, CYAN),
    (0.84, LIGHT_CYAN),
    (0.94, CYAN_WHITE),
    (1.00, (1.00, 1.00, 1.00)),
]

# ============================================================================
# GLOBAL DECAY — the ONE knob for neuron + connection fade speed
# ============================================================================
# Set this to control how fast neurons and connections fade after a spike.
# Lower = faster fadeout.  Set to 0 to disable decay entirely (instant on/off).
# This value is used by BOTH neurons and connections uniformly.
DECAY_MS = 12.0

# Connection rendering
CONN_ALPHA_POWER = 3.0     # exponential alpha curve (higher = more aggressive cutoff)
CONN_ALPHA_MAX = 50.0      # peak alpha for brightest connections

# Connection difference mode: highlights sudden activity changes
# When enabled, connections that are normally dormant but suddenly spike
# appear bright, while constantly active connections stay dim.
# EMA_ALPHA controls how fast the baseline adapts (lower = longer memory)
CONN_DIFF_MODE = False      # toggle with D key at runtime
CONN_EMA_ALPHA = 0.004       # EMA smoothing factor (0.02=slow adapt, 0.1=fast)
CONN_DIFF_BOOST = 3.0       # amplification of the difference signal

# Splat scale multipliers (relative to world_unit)
AURA_SCALE_MULT = 0.72
BLOOM_SCALE_MULT = 1.34
CORE_SCALE_MULT = 0.35
HOT_SCALE_MULT = 0.22

# Opacity/color gains per layer
AURA_GAIN = 1.10
BLOOM_GAIN = 1.42
CORE_GAIN = 1.42
HOT_GAIN = 2.35

# ============================================================================
# WHITE PEAK GAIN — controls how easily neurons reach maximum white brightness
# ============================================================================
# Range: 0.3 (very rare white, most neurons stay orange/yellow)
#    to  1.0 (original behavior, many neurons go white)
#    to  1.5 (even more white)
# Adjust this to taste. 0.5 is a good starting point for "rare white".
WHITE_GAIN = 0.32

# Color ramp knees
WHITE_KNEE = 0.89
BLOOM_KNEE = 0.685
AURA_CENTER_RELIEF = 0.08

# Base point appearance
BASE_POINT_SIZE = 2.7
BASE_OFF_COLOR = np.array([0.07, 0.02, 0.006], dtype=np.float32)

# Base point suppression thresholds
BASE_HIDE_START = 0.26
BASE_HIDE_END = 0.42
BASE_HARD_HIDE = 0.48
BASE_KEEP_EPS = 0.012

# Density-adaptive world_unit scaling constant
# world_unit = diag / (n^(1/3) * DENSITY_K)
# Tuned so 356 neurons gives the same scale as the old diag/125 formula
DENSITY_K = 17.6


# ============================================================================
# VTK Helpers
# ============================================================================

def _maybe_call(obj, method, *args):
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args)
    return None


def _make_verts_fast(n):
    """Build vertex cell array using numpy (much faster than Python loop for large n)."""
    n = int(n)
    # VTK cell array format: [1, 0, 1, 1, 1, 2, ...] = [npts, pt_id, npts, pt_id, ...]
    conn = np.empty(2 * n, dtype=np.int64)
    conn[0::2] = 1           # each cell has 1 point
    conn[1::2] = np.arange(n, dtype=np.int64)
    offsets = np.arange(0, 2 * n + 1, 2, dtype=np.int64)

    verts = vtk.vtkCellArray()
    try:
        # VTK 9+ fast path
        conn_vtk = numpy_support.numpy_to_vtk(conn, deep=True,
                                                array_type=vtk.VTK_ID_TYPE)
        off_vtk = numpy_support.numpy_to_vtk(offsets, deep=True,
                                               array_type=vtk.VTK_ID_TYPE)
        verts.SetData(off_vtk, conn_vtk)
    except (TypeError, AttributeError):
        # Fallback for older VTK
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
    return verts


def _set_point_polydata(poly, pts, rgba):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(pts, deep=True))
    colors = numpy_support.numpy_to_vtk(rgba, deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetNumberOfComponents(4)
    colors.SetName("colors")
    poly.SetPoints(vtk_pts)
    poly.SetVerts(_make_verts_fast(len(pts)))
    poly.GetPointData().SetScalars(colors)
    poly.Modified()


def _set_point_polydata_colors_only(poly, rgba):
    """Update only the color array of existing polydata (avoids rebuilding geometry)."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    colors = numpy_support.numpy_to_vtk(rgba, deep=True,
                                         array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetNumberOfComponents(4)
    colors.SetName("colors")
    poly.GetPointData().SetScalars(colors)
    poly.Modified()


def _make_splat_polydata(pts, activity):
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(pts, dtype=np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetVerts(_make_verts_fast(len(pts)))
    act_vtk = numpy_support.numpy_to_vtk(activity, deep=False)
    act_vtk.SetName("activity")
    poly.GetPointData().AddArray(act_vtk)
    poly.GetPointData().SetActiveScalars("activity")
    return poly, act_vtk


def _build_fire_color_lut(stops, n=256):
    """Build a numpy LUT (n, 3) from color ramp stops for fast per-line lookup."""
    lut = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        t = i / max(n - 1, 1)
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                span = stops[j + 1][0] - stops[j][0]
                frac = (t - stops[j][0]) / max(span, 1e-8)
                c0 = np.array(stops[j][1], dtype=np.float32)
                c1 = np.array(stops[j + 1][1], dtype=np.float32)
                lut[i] = c0 * (1.0 - frac) + c1 * frac
                break
    return lut


def _make_color_tf(stops):
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToRGB()
    for x, (r, g, b) in stops:
        ctf.AddRGBPoint(float(x), float(r), float(g), float(b))
    return ctf


def _make_opacity_tf(stops, gain=1.0):
    otf = vtk.vtkPiecewiseFunction()
    for x, a in stops:
        otf.AddPoint(float(x), float(np.clip(a * gain, 0.0, 1.0)))
    return otf


def _splat_shader(kind):
    if kind == "aura":
        return f"""
        //VTK::Color::Impl
        float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
        if (r2 > 1.0) {{ discard; }}
        float r = sqrt(r2);
        float broad = exp(-1.18 * r2);
        float mid = exp(-4.75 * r2);
        float notch = exp(-24.0 * r2);
        float centerRelief = 1.0 - {AURA_CENTER_RELIEF:.5f} * notch;
        float edge = 1.0 - smoothstep(0.96, 1.00, r);
        float shape = (0.76 * broad + 0.34 * mid) * centerRelief * edge;
        diffuseColor *= shape;
        ambientColor *= shape;
        opacity *= clamp(shape, 0.0, 1.0);
        if (opacity < 0.0015) {{ discard; }}
        """
    if kind == "bloom":
        return """
        //VTK::Color::Impl
        float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
        if (r2 > 1.0) { discard; }
        float r = sqrt(r2);
        float broad = exp(-0.72 * r2);
        float shoulder = exp(-2.8 * r2);
        float skirt = exp(-8.5 * r2);
        float edge = 1.0 - smoothstep(0.975, 1.00, r);
        float shape = (0.66 * broad + 0.26 * shoulder + 0.12 * skirt) * edge;
        diffuseColor *= shape;
        ambientColor *= shape;
        opacity *= clamp(shape, 0.0, 1.0);
        if (opacity < 0.0010) { discard; }
        """
    if kind == "core":
        return """
        //VTK::Color::Impl
        float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
        if (r2 > 1.0) { discard; }
        float r = sqrt(r2);
        float g0 = exp(-4.0 * r2);
        float g1 = exp(-14.0 * r2);
        float edge = 1.0 - smoothstep(0.965, 1.00, r);
        float shape = (0.64 * g0 + 0.56 * g1) * edge;
        diffuseColor *= shape;
        ambientColor *= shape;
        opacity *= clamp(shape, 0.0, 1.0);
        if (opacity < 0.0018) { discard; }
        """
    if kind == "hot":
        return """
        //VTK::Color::Impl
        float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);
        if (r2 > 1.0) { discard; }
        float r = sqrt(r2);
        float c0 = exp(-10.0 * r2);
        float c1 = exp(-44.0 * r2);
        float edge = 1.0 - smoothstep(0.96, 1.00, r);
        float shape = (0.20 * c0 + 1.82 * c1) * edge;
        diffuseColor *= shape;
        ambientColor *= shape;
        opacity *= clamp(shape, 0.0, 1.0);
        if (opacity < 0.0032) { discard; }
        """
    raise ValueError(f"Unknown splat kind: {kind}")


def _make_gaussian_actor(poly, kind, scale_factor, color_tf, opacity_tf):
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("activity")
    _maybe_call(mapper, "SetOpacityArray", "activity")
    _maybe_call(mapper, "SetOpacityArrayComponent", 0)
    _maybe_call(mapper, "SetScalarRange", 0.0, 1.0)
    _maybe_call(mapper, "UseLookupTableScalarRangeOn")
    _maybe_call(mapper, "SetLookupTable", color_tf)
    _maybe_call(mapper, "SetScalarOpacityFunction", opacity_tf)
    _maybe_call(mapper, "SetScaleFactor", float(scale_factor))
    _maybe_call(mapper, "SetSplatShaderCode", _splat_shader(kind))
    _maybe_call(mapper, "SetEmissive", 1)
    _maybe_call(mapper, "EmissiveOn")
    act = vtk.vtkActor()
    act.SetMapper(mapper)
    prop = act.GetProperty()
    prop.LightingOff()
    prop.SetOpacity(1.0)
    return act


def _make_base_actor(poly, point_size):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn()
    mapper.SetColorModeToDirectScalars()
    act = vtk.vtkActor()
    act.SetMapper(mapper)
    prop = act.GetProperty()
    prop.SetRepresentationToPoints()
    prop.SetPointSize(point_size)
    prop.LightingOff()
    try:
        prop.SetRenderPointsAsSpheres(False)
    except Exception:
        pass
    return act


# ============================================================================
# Transfer functions (from playground v10)
# ============================================================================

def _build_transfer_functions():
    aura_ctf = _make_color_tf([
        (0.00, BLACK), (0.04, DEEP_RED), (0.16, BLOOD_RED),
        (0.42, ORANGE_RED), (0.76, AMBER), (1.00, YELLOW),
    ])
    aura_otf = _make_opacity_tf([
        (0.00, 0.00), (0.04, 0.00), (0.12, 0.10),
        (0.34, 0.38), (0.68, 0.78), (1.00, 1.00),
    ], gain=AURA_GAIN)

    bloom_ctf = _make_color_tf([
        (0.00, BLACK), (BLOOM_KNEE - 0.06, BLACK), (BLOOM_KNEE, ORANGE_RED),
        (0.84, YELLOW), (0.95, PALE_YELLOW), (1.00, WARM_WHITE),
    ])
    bloom_otf = _make_opacity_tf([
        (0.00, 0.00), (BLOOM_KNEE - 0.07, 0.00), (BLOOM_KNEE, 0.08),
        (0.82, 0.30), (0.92, 0.72), (1.00, 1.00),
    ], gain=BLOOM_GAIN)

    core_ctf = _make_color_tf([
        (0.00, BLACK), (0.05, DEEP_RED), (0.21, BLOOD_RED),
        (0.50, ORANGE_RED), (0.78, YELLOW), (0.91, PALE_YELLOW),
        (1.00, WARM_WHITE),
    ])
    core_otf = _make_opacity_tf([
        (0.00, 0.00), (0.05, 0.00), (0.14, 0.12),
        (0.42, 0.56), (0.78, 0.96), (1.00, 1.00),
    ], gain=CORE_GAIN)

    hot_ctf = _make_color_tf([
        (0.00, BLACK), (WHITE_KNEE - 0.05, BLACK), (WHITE_KNEE, AMBER),
        (0.922, PALE_YELLOW), (0.948, WARM_WHITE),
        (0.974, (1.0, 1.0, 1.0)), (1.00, (1.0, 1.0, 1.0)),
    ])
    hot_otf = _make_opacity_tf([
        (0.00, 0.00), (WHITE_KNEE - 0.06, 0.00), (WHITE_KNEE, 0.28),
        (0.922, 0.86), (0.948, 1.00), (0.974, 1.00), (1.00, 1.00),
    ], gain=HOT_GAIN)

    return (aura_ctf, aura_otf, bloom_ctf, bloom_otf,
            core_ctf, core_otf, hot_ctf, hot_otf)


# ============================================================================
# BrainRenderer
# ============================================================================

class BrainRenderer:
    """GPU-accelerated 3D brain visualization with gaussian splat glow.

    Args:
        performance: if True, uses fewer splat layers and optimized update
                     loop for rendering 100k+ neurons in real-time.
    """

    def __init__(self, positions, connectivity=None, conn_weights=None,
                 conn_excitatory=None, skeletons=None,
                 subset_ids=None, neuron_ids=None,
                 performance=False):
        self.performance = performance
        self.scene = window.Scene()
        self.scene.background((0.0, 0.0, 0.0))  # pure black

        # Build ordered neuron list
        if neuron_ids is not None:
            self.neuron_ids = list(neuron_ids)
        else:
            self.neuron_ids = sorted(positions.keys())

        if subset_ids is not None:
            self.active_ids = [rid for rid in self.neuron_ids if rid in subset_ids]
        else:
            self.active_ids = self.neuron_ids

        self.id_to_idx = {rid: i for i, rid in enumerate(self.active_ids)}
        self.n_active = len(self.active_ids)

        # Extract positions
        self.positions_dict = positions
        self.pos_array = np.array([
            positions.get(rid, (0, 0, 0)) for rid in self.active_ids
        ], dtype=np.float64)

        # Normalize to rendering space
        if len(self.pos_array) > 0:
            self.center = self.pos_array.mean(axis=0)
            extent = self.pos_array.max(axis=0) - self.pos_array.min(axis=0)
            self.scale = max(extent.max(), 1.0)
            self.pos_normalized = ((self.pos_array - self.center) /
                                   self.scale * 200.0).astype(np.float32)
        else:
            self.center = np.zeros(3)
            self.scale = 1.0
            self.pos_normalized = np.zeros((0, 3), dtype=np.float32)

        # World unit: based on actual nearest-neighbor distance
        # This adapts perfectly to any density / clustering pattern
        if len(self.pos_normalized) > 1:
            bbox = self.pos_normalized.max(axis=0) - self.pos_normalized.min(axis=0)
            self.diag = float(np.linalg.norm(bbox))
        else:
            self.diag = 200.0

        self.world_unit = self._compute_world_unit()

        self.connectivity = connectivity
        self.conn_weights = conn_weights
        self.conn_excitatory = conn_excitatory
        self.skeletons = skeletons or {}

        # Rendering state
        self._activity = np.zeros(self.n_active, dtype=np.float32)
        self._splat_poly = None
        self._activity_vtk = None
        self._base_poly = None

        # Connection animation state
        self._conn_visible = True  # toggled with C key
        self._conn_actor = None
        self._conn_colors_vtk = None
        self._conn_n_lines = 0
        self._conn_pre_idx = None
        self._conn_post_idx = None
        self._conn_weight_norm = None
        self._conn_is_exc = None
        self._conn_verts_per_line = None
        self._conn_exc_lut = _build_fire_color_lut(CONN_EXC_STOPS, 256)
        self._conn_inh_lut = _build_fire_color_lut(CONN_INH_STOPS, 256)

        # Connection difference mode state
        self._conn_diff_mode = CONN_DIFF_MODE
        self._conn_ema = None  # allocated on first use (needs _conn_n_lines)

        # Neuron selection / inspection state
        self._selected_idx = None        # index into active_ids
        self._selected_partners = False  # True = showing partners
        self._select_actor = None        # highlight ring actor
        self._partner_actor = None       # partner highlight actor
        self._info_label = None          # info text overlay
        # Build per-neuron connectivity lookup for inspection
        self._neuron_pre = {}   # neuron_idx -> list of (post_idx, weight, is_exc)
        self._neuron_post = {}  # neuron_idx -> list of (pre_idx, weight, is_exc)

        # Performance mode: frame counter for skipping connection updates
        self._perf_frame = 0

        # Pre-allocate base point RGBA for in-place updates (avoids alloc per frame)
        self._base_rgba = np.zeros((self.n_active, 4), dtype=np.uint8)

        self._build_neurons()
        if connectivity is not None:
            self._build_connections(connectivity, conn_weights,
                                    conn_excitatory, subset_ids)

    def _compute_world_unit(self):
        """Compute world_unit from actual nearest-neighbor distances.

        Samples a subset of neurons and finds median nearest-neighbor distance,
        then scales splats to be a fraction of that spacing. This adapts
        perfectly to any neuron density / clustering pattern.
        """
        pts = self.pos_normalized
        n = len(pts)

        if n <= 1:
            return self.diag / 125.0

        # For large neuron counts, sample to keep this fast
        if n > 5000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, 3000, replace=False)
            sample_pts = pts[sample_idx]
        else:
            sample_pts = pts

        # Compute nearest-neighbor distance for each sampled point
        # Use scipy if available, else brute force on sample
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)  # build tree on ALL points
            dists, _ = tree.query(sample_pts, k=2)  # k=2: self + nearest
            nn_dists = dists[:, 1]  # second column is nearest neighbor
        except ImportError:
            # Brute force fallback on small sample
            if n > 2000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n, 500, replace=False)
                sample_pts = pts[idx]
            nn_dists = np.zeros(len(sample_pts), dtype=np.float32)
            for i, p in enumerate(sample_pts):
                d = np.linalg.norm(pts - p[None, :], axis=1)
                d[d == 0] = 1e9  # exclude self
                nn_dists[i] = d.min()

        median_nn = float(np.median(nn_dists))

        # world_unit = fraction of median nearest-neighbor distance
        # 0.6 means the core splat radius is ~60% of the typical spacing
        # This prevents overlapping halos while keeping visible glow
        wu = median_nn * 0.6

        # Clamp to reasonable range
        wu = max(wu, self.diag / 2000.0)  # don't go too tiny
        wu = min(wu, self.diag / 50.0)    # don't go too huge

        print(f"[visualizer] Nearest-neighbor median={median_nn:.3f}, "
              f"world_unit={wu:.4f}")

        return wu

    def _build_neurons(self):
        """Create neuron rendering actors: base points + gaussian splat layers."""
        if self.n_active == 0:
            return

        pts = self.pos_normalized
        n = self.n_active
        wu = self.world_unit

        # --- Base points (warm ember dots, suppressed when active) ---
        rng = np.random.default_rng(42)
        base_mix = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        off_0 = np.array([0.070, 0.020, 0.006], dtype=np.float32)
        off_1 = np.array([0.220, 0.078, 0.016], dtype=np.float32)
        self._base_rgb_off = off_0[None, :] * (1.0 - base_mix) + off_1[None, :] * base_mix
        self._base_alpha_off = (0.18 + 0.08 * base_mix[:, 0]).astype(np.float32)

        # For performance mode with large neuron counts, dim the base points
        if self.performance:
            self._base_alpha_off *= 0.5
            point_size = max(1.0, BASE_POINT_SIZE * 0.6)
        else:
            point_size = BASE_POINT_SIZE

        self._base_poly = vtk.vtkPolyData()
        base_rgba = np.zeros((n, 4), dtype=np.uint8)
        base_rgba[:, :3] = (self._base_rgb_off * 255).astype(np.uint8)
        base_rgba[:, 3] = (self._base_alpha_off * 255).astype(np.uint8)
        _set_point_polydata(self._base_poly, pts, base_rgba)
        self._base_actor = _make_base_actor(self._base_poly, point_size)
        self.scene.add(self._base_actor)

        # --- Gaussian splat layers ---
        self._activity = np.zeros(n, dtype=np.float32)
        self._splat_poly, self._activity_vtk = _make_splat_polydata(
            pts, self._activity)

        tfs = _build_transfer_functions()
        (aura_ctf, aura_otf, bloom_ctf, bloom_otf,
         core_ctf, core_otf, hot_ctf, hot_otf) = tfs

        if self.performance:
            # Performance mode: only core + hot layers (2 draw calls instead of 4)
            self._aura_actor = None
            self._bloom_actor = None
            self._core_actor = _make_gaussian_actor(
                self._splat_poly, "core", CORE_SCALE_MULT * wu, core_ctf, core_otf)
            self._hot_actor = _make_gaussian_actor(
                self._splat_poly, "hot", HOT_SCALE_MULT * wu, hot_ctf, hot_otf)
            self.scene.add(self._core_actor)
            self.scene.add(self._hot_actor)
            mode_str = "PERFORMANCE (core+hot)"
        else:
            # Full quality: all 4 layers
            self._aura_actor = _make_gaussian_actor(
                self._splat_poly, "aura", AURA_SCALE_MULT * wu, aura_ctf, aura_otf)
            self._bloom_actor = _make_gaussian_actor(
                self._splat_poly, "bloom", BLOOM_SCALE_MULT * wu, bloom_ctf, bloom_otf)
            self._core_actor = _make_gaussian_actor(
                self._splat_poly, "core", CORE_SCALE_MULT * wu, core_ctf, core_otf)
            self._hot_actor = _make_gaussian_actor(
                self._splat_poly, "hot", HOT_SCALE_MULT * wu, hot_ctf, hot_otf)
            self.scene.add(self._aura_actor)
            self.scene.add(self._bloom_actor)
            self.scene.add(self._core_actor)
            self.scene.add(self._hot_actor)
            mode_str = "FULL (aura+bloom+core+hot)"

        print(f"[visualizer] {n} neurons, mode={mode_str}, "
              f"world_unit={wu:.4f}, diag={self.diag:.1f}")

    def _build_connections(self, connectivity, conn_weights=None,
                           conn_excitatory=None, subset_ids=None):
        """Build connection line actor with per-line color animation support."""
        pre_ids, post_ids = connectivity

        # Filter to active neurons — vectorized
        active_set = set(self.id_to_idx.keys())
        pre_in = np.array([p in active_set for p in pre_ids], dtype=bool)
        post_in = np.array([p in active_set for p in post_ids], dtype=bool)
        mask = pre_in & post_in
        pre_ids = pre_ids[mask]
        post_ids = post_ids[mask]

        if conn_weights is not None:
            conn_weights = conn_weights[mask]
        if conn_excitatory is not None:
            conn_excitatory = conn_excitatory[mask]

        # Subsample if too many — cap per neuron to prevent hub domination
        n_conns = len(pre_ids)
        max_conns = 15000 if self.performance else 30000
        if n_conns > max_conns:
            # Per-neuron cap: each neuron contributes at most N connections
            # This prevents hub neurons from dominating the visual
            max_per_neuron = max(20, max_conns // max(len(active_set), 1))

            # Sort by weight descending to keep strongest connections
            if conn_weights is not None and len(conn_weights) > 0:
                weight_order = np.argsort(-conn_weights)
            else:
                weight_order = np.arange(n_conns)

            pre_counts = {}
            kept = []
            for idx in weight_order:
                pid = int(pre_ids[idx])
                c = pre_counts.get(pid, 0)
                if c < max_per_neuron:
                    kept.append(idx)
                    pre_counts[pid] = c + 1
                    if len(kept) >= max_conns:
                        break

            indices = np.array(kept, dtype=np.int64)
            pre_ids = pre_ids[indices]
            post_ids = post_ids[indices]
            if conn_weights is not None:
                conn_weights = conn_weights[indices]
            if conn_excitatory is not None:
                conn_excitatory = conn_excitatory[indices]
            n_sources = len(pre_counts)
            print(f"[visualizer] Connection sampling: {n_sources} source neurons, "
                  f"max {max_per_neuron}/neuron")

        n_lines = len(pre_ids)
        if n_lines == 0:
            return

        # Store mapping for animation
        self._conn_pre_idx = np.array(
            [self.id_to_idx[p] for p in pre_ids], dtype=np.int32)
        self._conn_post_idx = np.array(
            [self.id_to_idx[p] for p in post_ids], dtype=np.int32)

        # Normalize weights to 0..1 (log scale since distribution is heavy-tailed)
        if conn_weights is not None:
            w = np.log1p(conn_weights.astype(np.float32))
            w_max = w.max() if w.max() > 0 else 1.0
            self._conn_weight_norm = np.clip(w / w_max, 0.0, 1.0)
        else:
            self._conn_weight_norm = np.ones(n_lines, dtype=np.float32) * 0.5

        if conn_excitatory is not None:
            self._conn_is_exc = conn_excitatory > 0
        else:
            self._conn_is_exc = np.ones(n_lines, dtype=bool)

        self._conn_n_lines = n_lines

        # Build VTK line polydata — vectorized point assembly
        all_pts = np.zeros((n_lines * 2, 3), dtype=np.float32)
        all_pts[0::2] = self.pos_normalized[self._conn_pre_idx]
        all_pts[1::2] = self.pos_normalized[self._conn_post_idx]

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_support.numpy_to_vtk(all_pts, deep=True))

        # Build line cells — vectorized
        # Each line: [2, pt_a, pt_b]
        cell_conn = np.empty(3 * n_lines, dtype=np.int64)
        cell_conn[0::3] = 2
        cell_conn[1::3] = np.arange(0, 2 * n_lines, 2, dtype=np.int64)
        cell_conn[2::3] = np.arange(1, 2 * n_lines, 2, dtype=np.int64)
        cell_offsets = np.arange(0, 3 * n_lines + 1, 3, dtype=np.int64)

        lines = vtk.vtkCellArray()
        try:
            conn_vtk = numpy_support.numpy_to_vtk(cell_conn, deep=True,
                                                    array_type=vtk.VTK_ID_TYPE)
            off_vtk = numpy_support.numpy_to_vtk(cell_offsets, deep=True,
                                                   array_type=vtk.VTK_ID_TYPE)
            lines.SetData(off_vtk, conn_vtk)
        except (TypeError, AttributeError):
            for i in range(n_lines):
                lines.InsertNextCell(2)
                lines.InsertCellPoint(2 * i)
                lines.InsertCellPoint(2 * i + 1)

        # Initial colors: fully dark and transparent
        n_verts = n_lines * 2
        colors_rgba = np.zeros((n_verts, 4), dtype=np.uint8)

        self._conn_colors_vtk = numpy_support.numpy_to_vtk(
            colors_rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        self._conn_colors_vtk.SetNumberOfComponents(4)
        self._conn_colors_vtk.SetName("Colors")

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetLines(lines)
        poly.GetPointData().SetScalars(self._conn_colors_vtk)

        self._conn_poly = poly

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOn()
        mapper.SetColorModeToDirectScalars()

        self._conn_actor = vtk.vtkActor()
        self._conn_actor.SetMapper(mapper)
        prop = self._conn_actor.GetProperty()
        prop.SetLineWidth(1.0)
        prop.LightingOff()
        prop.SetOpacity(1.0)
        # Enable line antialiasing (smooth thin lines)
        try:
            prop.SetRenderLinesAsTubes(False)
            prop.EdgeVisibilityOff()
        except Exception:
            pass

        self.scene.add(self._conn_actor)

        # Build per-neuron connectivity lookup for neuron inspection
        for i in range(n_lines):
            pre = int(self._conn_pre_idx[i])
            post = int(self._conn_post_idx[i])
            w = float(self._conn_weight_norm[i])
            exc = bool(self._conn_is_exc[i]) if self._conn_is_exc is not None else True
            self._neuron_pre.setdefault(pre, []).append((post, w, exc))
            self._neuron_post.setdefault(post, []).append((pre, w, exc))

        print(f"[visualizer] Rendered {n_lines} connections with weight animation")

    def _build_hud(self):
        """Build heads-up display text overlays."""
        mode_str = " [PERF]" if self.performance else ""
        self.time_label = ui.TextBlock2D(
            text="t = 0.0 ms",
            position=(20, 20),
            font_size=18,
            color=(0.8, 0.6, 0.3),
            bold=True,
        )
        self.info_label = ui.TextBlock2D(
            text=f"Neurons: {self.n_active:,}{mode_str}",
            position=(20, 50),
            font_size=14,
            color=(0.5, 0.35, 0.2),
        )
        self.controls_label = ui.TextBlock2D(
            text="[Space] Pause  [R] Reset  [+/-] Speed  [C] Connections  [D] Diff  [Q] Quit",
            position=(20, 80),
            font_size=12,
            color=(0.3, 0.2, 0.15),
        )
        self._select_label = ui.TextBlock2D(
            text="",
            position=(20, 110),
            font_size=13,
            color=(0.9, 0.75, 0.4),
        )
        self.scene.add(self.time_label)
        self.scene.add(self.info_label)
        self.scene.add(self.controls_label)
        self.scene.add(self._select_label)

    def _find_nearest_neuron(self, click_pos_3d):
        """Find the neuron index nearest to a 3D world position."""
        if self.n_active == 0:
            return None
        dists = np.linalg.norm(
            self.pos_normalized - np.array(click_pos_3d, dtype=np.float32),
            axis=1)
        idx = int(np.argmin(dists))
        # Only select if reasonably close (within 3x world_unit)
        if dists[idx] > self.world_unit * 8.0:
            return None
        return idx

    def _select_neuron(self, idx):
        """Select or deselect a neuron. First click = select + info,
        second click on same = show partners, third = deselect."""
        if idx is None:
            self._deselect()
            return

        if self._selected_idx == idx:
            if not self._selected_partners:
                # Second click: show partners
                self._show_partners(idx)
                return
            else:
                # Third click: deselect
                self._deselect()
                return

        # New selection
        self._deselect()
        self._selected_idx = idx
        self._selected_partners = False

        # Create highlight ring around selected neuron
        pos = self.pos_normalized[idx:idx + 1]
        rgba = np.array([[255, 220, 60, 220]], dtype=np.uint8)
        self._select_poly = vtk.vtkPolyData()
        _set_point_polydata(self._select_poly, pos, rgba)
        self._select_actor = _make_base_actor(self._select_poly, 12.0)
        self.scene.add(self._select_actor)

        # Show info
        rid = self.active_ids[idx]
        n_out = len(self._neuron_pre.get(idx, []))
        n_in = len(self._neuron_post.get(idx, []))
        exc_out = sum(1 for _, _, e in self._neuron_pre.get(idx, []) if e)
        inh_out = n_out - exc_out
        act = self._activity[idx] if idx < len(self._activity) else 0
        info = (f"Neuron {rid}  |  out: {n_out} ({exc_out}E/{inh_out}I)  "
                f"in: {n_in}  |  activity: {act:.2f}  "
                f"[click again: show partners]")
        self._select_label.message = info
        print(f"[select] {info}")

    def _show_partners(self, idx):
        """Highlight all pre/post-synaptic partners of selected neuron."""
        self._selected_partners = True

        partner_indices = set()
        for post, w, exc in self._neuron_pre.get(idx, []):
            partner_indices.add(post)
        for pre, w, exc in self._neuron_post.get(idx, []):
            partner_indices.add(pre)

        if not partner_indices:
            self._select_label.message += "  (no partners in view)"
            return

        partner_list = sorted(partner_indices)
        pos = self.pos_normalized[partner_list]

        # Color partners: green for post-synaptic targets, orange for inputs
        pre_set = {pre for pre, _, _ in self._neuron_post.get(idx, [])}
        rgba = np.zeros((len(partner_list), 4), dtype=np.uint8)
        for i, pidx in enumerate(partner_list):
            if pidx in pre_set:
                rgba[i] = [60, 200, 255, 180]   # cyan = input to this neuron
            else:
                rgba[i] = [100, 255, 80, 180]    # green = target of this neuron

        self._partner_poly = vtk.vtkPolyData()
        _set_point_polydata(self._partner_poly, pos, rgba)
        self._partner_actor = _make_base_actor(self._partner_poly, 8.0)
        self.scene.add(self._partner_actor)

        n_post = sum(1 for p in partner_list if p not in pre_set)
        n_pre = len(partner_list) - n_post
        self._select_label.message = (
            f"Neuron {self.active_ids[idx]}  |  "
            f"showing {n_pre} inputs (cyan) + {n_post} targets (green)  "
            f"[click again: deselect]")

    def _deselect(self):
        """Remove selection highlight."""
        self._selected_idx = None
        self._selected_partners = False
        if self._select_actor is not None:
            self.scene.rm(self._select_actor)
            self._select_actor = None
        if self._partner_actor is not None:
            self.scene.rm(self._partner_actor)
            self._partner_actor = None
        if self._select_label is not None:
            self._select_label.message = ""

    def update_activity(self, brightness, conn_brightness=None):
        """Update neuron activity and connection flash based on brightness.

        Args:
            brightness: array of shape (n_active,) with values 0..1
            conn_brightness: optional separate brightness for connections
                             (with faster decay). Falls back to brightness.
        """
        if self.n_active == 0:
            return

        b = np.clip(brightness[:self.n_active].astype(np.float32), 0.0, 1.0)

        # Apply intensity shaping (from playground v10)
        # WHITE_GAIN scales input brightness before shaping —
        # lower values compress the range so fewer neurons reach white
        b_scaled = np.clip(b * WHITE_GAIN, 0.0, 1.0)
        saturated = 1.0 - np.exp(-3.3 * b_scaled)
        shoulder = 1.0 - np.exp(-2.25 * np.maximum(saturated - 0.10, 0.0))
        combined = np.clip(0.64 * saturated + 0.82 * shoulder, 0.0, 1.0)
        shaped = np.power(combined, 0.62)
        shaped = np.clip(
            shaped + 0.22 * np.power(np.clip(combined - 0.72, 0.0, 1.0), 0.45),
            0.0, 1.0)
        self._activity[:] = shaped

        # Update splat activity values
        if self._activity_vtk is not None:
            arr = numpy_support.vtk_to_numpy(self._activity_vtk)
            arr[:] = self._activity
            self._activity_vtk.Modified()
            self._splat_poly.Modified()

        # Update base points
        if self._base_poly is not None:
            if self.performance:
                # Performance mode: skip per-frame base point rebuilds
                # Just update colors in-place (no geometry rebuild)
                self._update_base_points_fast()
            else:
                # Full mode: fancy suppression with geometry filtering
                self._update_base_points_full()

        # Update connection colors
        self._perf_frame += 1
        # In performance mode, update connections every other frame
        if not self.performance or (self._perf_frame % 2 == 0):
            cb = conn_brightness if conn_brightness is not None else brightness
            cb = np.clip(cb[:self.n_active].astype(np.float32), 0.0, 1.0)
            self._update_connection_colors(cb)

    def _update_base_points_fast(self):
        """Performance mode: update base point colors without rebuilding geometry."""
        a = self._activity
        # Simple fade: when active, dim the base point
        gate = np.clip(1.0 - a * 2.5, 0.0, 1.0)

        rgba = self._base_rgba
        rgba[:, :3] = np.clip(self._base_rgb_off * gate[:, None] * 255.0,
                              0, 255).astype(np.uint8)
        rgba[:, 3] = np.clip(self._base_alpha_off * gate * 255.0,
                             0, 255).astype(np.uint8)
        _set_point_polydata_colors_only(self._base_poly, rgba)

    def _update_base_points_full(self):
        """Full mode: suppress base points with smoothstep when splats are visible."""
        a = self._activity
        t_self = np.clip((a - BASE_HIDE_START) /
                         max(BASE_HIDE_END - BASE_HIDE_START, 1e-8), 0, 1)
        gate = 1.0 - t_self * t_self * (3.0 - 2.0 * t_self)
        gate[a >= BASE_HARD_HIDE] = 0.0
        keep = gate > BASE_KEEP_EPS

        if np.any(keep):
            vis = np.power(gate[keep], 2.4)
            heat = np.power(a[keep], 1.15)
            ember = np.array([1.0, 0.62, 0.18], dtype=np.float32)
            rgb = self._base_rgb_off[keep] * (0.98 * vis[:, None])
            rgb += ember[None, :] * (0.08 * heat[:, None] * vis[:, None])
            alpha = self._base_alpha_off[keep] * vis

            rgba = np.zeros((keep.sum(), 4), dtype=np.uint8)
            rgba[:, :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            rgba[:, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            _set_point_polydata(
                self._base_poly, self.pos_normalized[keep], rgba)
        else:
            _set_point_polydata(
                self._base_poly,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 4), dtype=np.uint8))

    def _update_connection_colors(self, conn_brightness):
        """Flash connections using color ramp + exponential alpha.

        Supports two modes:
        - Normal: brightness = pre-synaptic activity * weight
        - Difference (D key): brightness = current - running_average,
          so dormant connections that suddenly spike appear bright
        """
        if self._conn_actor is None or self._conn_n_lines == 0:
            return

        # Connection intensity = pre-synaptic brightness * weight
        # with slight post-synaptic contribution for signal propagation feel
        pre_b = conn_brightness[self._conn_pre_idx]
        post_b = conn_brightness[self._conn_post_idx]
        raw = pre_b * self._conn_weight_norm * 0.85 + post_b * 0.15

        # Difference mode: highlight sudden changes over baseline
        if self._conn_diff_mode:
            if self._conn_ema is None:
                self._conn_ema = raw.copy()
            else:
                # Update exponential moving average
                self._conn_ema += CONN_EMA_ALPHA * (raw - self._conn_ema)
            # Difference = how much current exceeds baseline, boosted
            diff = np.clip((raw - self._conn_ema) * CONN_DIFF_BOOST, 0.0, 1.0)
            raw = diff
        else:
            # Still update EMA so switching modes is smooth
            if self._conn_ema is None:
                self._conn_ema = raw.copy()
            else:
                self._conn_ema += CONN_EMA_ALPHA * (raw - self._conn_ema)

        # Apply same intensity shaping as neurons for consistent look
        saturated = 1.0 - np.exp(-3.3 * raw)
        shoulder = 1.0 - np.exp(-2.25 * np.maximum(saturated - 0.10, 0.0))
        combined = np.clip(0.64 * saturated + 0.82 * shoulder, 0.0, 1.0)
        shaped = np.power(combined, 0.62)
        shaped = np.clip(
            shaped + 0.22 * np.power(
                np.clip(combined - 0.72, 0.0, 1.0), 0.45),
            0.0, 1.0)

        # Sample color ramp via LUT — excitatory (violet) vs inhibitory (cyan)
        lut_idx = np.clip((shaped * 255).astype(np.int32), 0, 255)
        line_rgb = self._conn_exc_lut[lut_idx]  # default: excitatory (violet)
        if self._conn_is_exc is not None:
            inh_mask = ~self._conn_is_exc
            if np.any(inh_mask):
                line_rgb[inh_mask] = self._conn_inh_lut[lut_idx[inh_mask]]

        # Exponential alpha: power curve makes low-intensity connections
        # nearly invisible while bright ones pop
        line_alpha = np.power(shaped, CONN_ALPHA_POWER) * CONN_ALPHA_MAX

        # Apply to vertex colors (2 verts per line, same color for both)
        colors_np = numpy_support.vtk_to_numpy(self._conn_colors_vtk)
        rgb_u8 = np.clip(line_rgb * 255, 0, 255).astype(np.uint8)
        alpha_u8 = np.clip(line_alpha, 0, 255).astype(np.uint8)
        colors_np[0::2, :3] = rgb_u8
        colors_np[1::2, :3] = rgb_u8
        colors_np[0::2, 3] = alpha_u8
        colors_np[1::2, 3] = alpha_u8

        self._conn_colors_vtk.Modified()
        self._conn_poly.Modified()

    def update_hud(self, time_ms, speed, is_playing, duration_ms):
        """Update HUD text."""
        status = "Playing" if is_playing else "PAUSED"
        pct = (time_ms / duration_ms * 100) if duration_ms > 0 else 0
        self.time_label.message = (
            f"t = {time_ms:.1f} ms / {duration_ms:.0f} ms  "
            f"({pct:.0f}%)  [{status}]  Speed: {speed:.2f}x"
        )

    def start(self, player):
        """Launch the interactive rendering window with animation."""
        size = (1600, 900)
        show_manager = window.ShowManager(
            scene=self.scene,
            size=size,
            title="Drosophila Brain - Neural Activity Visualizer",
            reset_camera=False,
            order_transparent=True,
        )

        # Camera setup
        cam_z = self.diag * 1.55
        self.scene.set_camera(
            position=(0.0, 0.0, cam_z),
            focal_point=(0.0, 0.0, 0.0),
            view_up=(0.0, 1.0, 0.0),
        )

        # Add HUD after render window exists
        self._build_hud()

        # Depth peeling for transparency + line antialiasing
        try:
            show_manager.window.SetAlphaBitPlanes(1)
            # Use multisampling for line AA (4x MSAA)
            show_manager.window.SetMultiSamples(4)
        except Exception:
            pass
        try:
            # Enable line smoothing at OpenGL level
            ren = show_manager.scene.GetRenderer() if hasattr(
                show_manager.scene, 'GetRenderer') else None
            if ren:
                ren.SetUseFXAA(True)
        except Exception:
            pass

        # Pre-build brightness-to-active mapping
        player_to_active = {}
        for rid, active_idx in self.id_to_idx.items():
            player_idx = player.neuron_index.get(rid, -1)
            if player_idx >= 0:
                player_to_active[player_idx] = active_idx

        player_indices = np.array(list(player_to_active.keys()), dtype=np.int32)
        active_indices = np.array(list(player_to_active.values()), dtype=np.int32)

        # Timer interval: 16ms (60fps) normal, 33ms (30fps) performance
        timer_ms = 33 if self.performance else 16

        # Animation callback
        _frame_count = [0]

        def timer_callback(_obj, _event):
            brightness = player.step()

            # Connection brightness: same decay as neurons (global DECAY_MS)
            conn_brightness_raw = player.get_brightness()

            # Map player brightness to active neurons
            active_brightness = np.zeros(self.n_active, dtype=np.float64)
            active_conn_brightness = np.zeros(self.n_active, dtype=np.float64)
            if len(player_indices) > 0:
                valid = player_indices < len(brightness)
                active_brightness[active_indices[valid]] = \
                    brightness[player_indices[valid]]
                active_conn_brightness[active_indices[valid]] = \
                    conn_brightness_raw[player_indices[valid]]

            _frame_count[0] += 1
            if _frame_count[0] % 60 == 1:
                n_bright = np.sum(active_brightness > 0.01)
                max_b = active_brightness.max() if self.n_active > 0 else 0
                print(f"[frame {_frame_count[0]}] t={player.current_time_ms:.1f}ms "
                      f"bright_neurons={n_bright} max_brightness={max_b:.3f}")

            self.update_activity(active_brightness,
                                 conn_brightness=active_conn_brightness)
            self.update_hud(
                player.current_time_ms,
                player.playback_speed,
                player.is_playing,
                player.duration_ms,
            )
            show_manager.render()

        # Keyboard controls
        def key_callback(obj, event):
            key = obj.GetKeySym()
            if key == 'space':
                player.toggle_pause()
            elif key in ('r', 'R'):
                player.reset()
            elif key in ('plus', 'equal'):
                player.set_speed(player.playback_speed * 1.5)
            elif key in ('minus', 'underscore'):
                player.set_speed(max(0.01, player.playback_speed / 1.5))
            elif key in ('c', 'C'):
                if self._conn_actor is not None:
                    self._conn_visible = not self._conn_visible
                    self._conn_actor.SetVisibility(self._conn_visible)
                    state = "ON" if self._conn_visible else "OFF"
                    print(f"[visualizer] Connections: {state}")
            elif key in ('d', 'D'):
                self._conn_diff_mode = not self._conn_diff_mode
                mode = "DIFF" if self._conn_diff_mode else "NORMAL"
                print(f"[visualizer] Connection mode: {mode}")
            elif key in ('q', 'Q'):
                show_manager.exit()

        # Click-to-select neuron (distinguish click from drag)
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.012)
        _mouse_press_pos = [None]

        def mouse_press_callback(obj, event):
            _mouse_press_pos[0] = obj.GetEventPosition()

        def mouse_release_callback(obj, event):
            release_pos = obj.GetEventPosition()
            press_pos = _mouse_press_pos[0]
            # Only treat as a click if mouse didn't move much (not a drag)
            if press_pos is not None:
                dx = abs(release_pos[0] - press_pos[0])
                dy = abs(release_pos[1] - press_pos[1])
                if dx < 5 and dy < 5:
                    # This was a click, not a drag
                    ren = show_manager.scene
                    picker.Pick(release_pos[0], release_pos[1], 0, ren)
                    pos = picker.GetPickPosition()
                    if pos == (0.0, 0.0, 0.0) and picker.GetPointId() < 0:
                        self._select_neuron(None)
                    else:
                        idx = self._find_nearest_neuron(pos)
                        self._select_neuron(idx)
            _mouse_press_pos[0] = None

        # Register callbacks
        show_manager.add_timer_callback(True, timer_ms, timer_callback)
        iren = show_manager.window.GetInteractor()
        iren.SetPicker(picker)
        iren.AddObserver('KeyPressEvent', key_callback)
        iren.AddObserver('LeftButtonPressEvent', mouse_press_callback)
        iren.AddObserver('LeftButtonReleaseEvent', mouse_release_callback)

        mode_str = " (PERFORMANCE MODE)" if self.performance else ""
        print(f"[visualizer] Window opened{mode_str}. Controls:")
        print("  Space:  Pause/Resume")
        print("  R:      Reset playback")
        print("  +/-:    Speed up/down")
        print("  C:      Toggle connections on/off")
        print("  D:      Toggle connection diff mode (highlights sudden changes)")
        print("  Click:  Select neuron (click again for partners, again to deselect)")

        show_manager.start()
