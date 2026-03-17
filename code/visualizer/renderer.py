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

# Connection flash colors
CONN_COLOR_REST = np.array([0.015, 0.025, 0.06], dtype=np.float32)
CONN_COLOR_EXC = np.array([1.0, 0.5, 0.05], dtype=np.float32)   # warm orange for excitatory
CONN_COLOR_INH = np.array([0.1, 0.4, 1.0], dtype=np.float32)    # blue for inhibitory

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


# ============================================================================
# VTK Helpers
# ============================================================================

def _maybe_call(obj, method, *args):
    fn = getattr(obj, method, None)
    if callable(fn):
        return fn(*args)
    return None


def _make_verts(n):
    verts = vtk.vtkCellArray()
    for i in range(int(n)):
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
    poly.SetVerts(_make_verts(len(pts)))
    poly.GetPointData().SetScalars(colors)
    poly.Modified()


def _make_splat_polydata(pts, activity):
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(pts, dtype=np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetVerts(_make_verts(len(pts)))
    act_vtk = numpy_support.numpy_to_vtk(activity, deep=False)
    act_vtk.SetName("activity")
    poly.GetPointData().AddArray(act_vtk)
    poly.GetPointData().SetActiveScalars("activity")
    return poly, act_vtk


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
    """GPU-accelerated 3D brain visualization with gaussian splat glow."""

    def __init__(self, positions, connectivity=None, conn_weights=None,
                 conn_excitatory=None, skeletons=None,
                 subset_ids=None, neuron_ids=None):
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

        # World unit for scale calculations
        if len(self.pos_normalized) > 1:
            bbox = self.pos_normalized.max(axis=0) - self.pos_normalized.min(axis=0)
            self.diag = float(np.linalg.norm(bbox))
        else:
            self.diag = 200.0
        self.world_unit = self.diag / 125.0

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
        self._conn_actor = None
        self._conn_colors_vtk = None
        self._conn_n_lines = 0
        self._conn_pre_idx = None
        self._conn_post_idx = None
        self._conn_weight_norm = None
        self._conn_is_exc = None
        self._conn_verts_per_line = None

        self._build_neurons()
        if connectivity is not None:
            self._build_connections(connectivity, conn_weights,
                                    conn_excitatory, subset_ids)

    def _build_neurons(self):
        """Create neuron rendering actors: base points + 4 gaussian splat layers."""
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

        self._base_poly = vtk.vtkPolyData()
        base_rgba = np.zeros((n, 4), dtype=np.uint8)
        base_rgba[:, :3] = (self._base_rgb_off * 255).astype(np.uint8)
        base_rgba[:, 3] = (self._base_alpha_off * 255).astype(np.uint8)
        _set_point_polydata(self._base_poly, pts, base_rgba)
        self._base_actor = _make_base_actor(self._base_poly, BASE_POINT_SIZE)
        self.scene.add(self._base_actor)

        # --- Gaussian splat layers ---
        self._activity = np.zeros(n, dtype=np.float32)
        self._splat_poly, self._activity_vtk = _make_splat_polydata(
            pts, self._activity)

        tfs = _build_transfer_functions()
        (aura_ctf, aura_otf, bloom_ctf, bloom_otf,
         core_ctf, core_otf, hot_ctf, hot_otf) = tfs

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

        print(f"[visualizer] {n} neurons with gaussian splat glow "
              f"(world_unit={wu:.3f})")

    def _build_connections(self, connectivity, conn_weights=None,
                           conn_excitatory=None, subset_ids=None):
        """Build connection line actor with per-line color animation support."""
        pre_ids, post_ids = connectivity

        # Filter to active neurons
        active_set = set(self.id_to_idx.keys())
        mask = np.array([
            pre in active_set and post in active_set
            for pre, post in zip(pre_ids, post_ids)
        ])
        pre_ids = pre_ids[mask]
        post_ids = post_ids[mask]

        # Keep weights/excitatory aligned
        if conn_weights is not None:
            conn_weights = conn_weights[mask]
        if conn_excitatory is not None:
            conn_excitatory = conn_excitatory[mask]

        # Subsample if too many
        n_conns = len(pre_ids)
        max_conns = 30000
        if n_conns > max_conns:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_conns, max_conns, replace=False)
            pre_ids = pre_ids[indices]
            post_ids = post_ids[indices]
            if conn_weights is not None:
                conn_weights = conn_weights[indices]
            if conn_excitatory is not None:
                conn_excitatory = conn_excitatory[indices]

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

        # Build VTK line polydata manually for per-vertex color control
        all_pts = np.zeros((n_lines * 2, 3), dtype=np.float32)
        for i in range(n_lines):
            all_pts[2 * i] = self.pos_normalized[self._conn_pre_idx[i]]
            all_pts[2 * i + 1] = self.pos_normalized[self._conn_post_idx[i]]

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_support.numpy_to_vtk(all_pts, deep=True))

        lines = vtk.vtkCellArray()
        for i in range(n_lines):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * i)
            lines.InsertCellPoint(2 * i + 1)

        # Initial resting colors (RGBA)
        rest_u8 = (CONN_COLOR_REST * 255).astype(np.uint8)
        n_verts = n_lines * 2
        colors_rgba = np.zeros((n_verts, 4), dtype=np.uint8)
        colors_rgba[:, :3] = rest_u8
        colors_rgba[:, 3] = 15  # low resting opacity

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
        prop.SetOpacity(1.0)  # per-vertex alpha controls opacity

        self.scene.add(self._conn_actor)
        print(f"[visualizer] Rendered {n_lines} connections with weight animation")

    def _build_hud(self):
        """Build heads-up display text overlays."""
        self.time_label = ui.TextBlock2D(
            text="t = 0.0 ms",
            position=(20, 20),
            font_size=18,
            color=(0.8, 0.6, 0.3),
            bold=True,
        )
        self.info_label = ui.TextBlock2D(
            text=f"Neurons: {self.n_active:,}",
            position=(20, 50),
            font_size=14,
            color=(0.5, 0.35, 0.2),
        )
        self.controls_label = ui.TextBlock2D(
            text="[Space] Pause  [R] Reset  [+/-] Speed  [Q] Quit",
            position=(20, 80),
            font_size=12,
            color=(0.3, 0.2, 0.15),
        )
        self.scene.add(self.time_label)
        self.scene.add(self.info_label)
        self.scene.add(self.controls_label)

    def update_activity(self, brightness):
        """Update neuron activity and connection flash based on brightness.

        Args:
            brightness: array of shape (n_active,) with values 0..1
        """
        if self.n_active == 0:
            return

        b = np.clip(brightness[:self.n_active].astype(np.float32), 0.0, 1.0)

        # Apply intensity shaping (from playground v10)
        # This makes flashes more punchy with sharper rise
        saturated = 1.0 - np.exp(-3.3 * b)
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

        # Update base points (suppress when active)
        if self._base_poly is not None:
            a = self._activity
            # Smoothstep suppression
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

        # Update connection colors (signal propagation)
        self._update_connection_colors(b)

    def _update_connection_colors(self, brightness):
        """Flash connections based on presynaptic neuron activity * weight."""
        if self._conn_actor is None or self._conn_n_lines == 0:
            return

        # Connection intensity = pre-neuron brightness * normalized weight
        pre_b = brightness[self._conn_pre_idx]
        post_b = brightness[self._conn_post_idx]

        # Signal propagation: connection glows based on pre-synaptic activity
        # with slight contribution from post-synaptic (receiving end)
        conn_intensity = pre_b * self._conn_weight_norm * 0.85 + post_b * 0.15
        conn_intensity = np.clip(conn_intensity, 0.0, 1.0)

        # Color: lerp from rest to active color based on intensity
        n = self._conn_n_lines
        rest = CONN_COLOR_REST
        exc_color = CONN_COLOR_EXC
        inh_color = CONN_COLOR_INH

        # Per-line target color
        target = np.where(
            self._conn_is_exc[:, None],
            exc_color[None, :],
            inh_color[None, :]
        )

        ci = conn_intensity[:, None]  # (n_lines, 1)
        line_rgb = rest[None, :] * (1.0 - ci) + target * ci

        # Alpha ramps from ~15 (rest) to 200 (active)
        line_alpha = 15.0 + conn_intensity * 185.0

        # Apply to vertex colors (2 verts per line, same color for both)
        colors_np = numpy_support.vtk_to_numpy(self._conn_colors_vtk)
        colors_np[0::2, :3] = np.clip(line_rgb * 255, 0, 255).astype(np.uint8)
        colors_np[1::2, :3] = np.clip(line_rgb * 255, 0, 255).astype(np.uint8)
        colors_np[0::2, 3] = np.clip(line_alpha, 0, 255).astype(np.uint8)
        colors_np[1::2, 3] = np.clip(line_alpha, 0, 255).astype(np.uint8)

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

        # Depth peeling for transparency
        try:
            show_manager.window.SetAlphaBitPlanes(1)
            show_manager.window.SetMultiSamples(0)
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

        # Animation callback
        _frame_count = [0]

        def timer_callback(_obj, _event):
            brightness = player.step()

            # Map player brightness to active neurons
            active_brightness = np.zeros(self.n_active, dtype=np.float64)
            if len(player_indices) > 0:
                valid = player_indices < len(brightness)
                active_brightness[active_indices[valid]] = \
                    brightness[player_indices[valid]]

            _frame_count[0] += 1
            if _frame_count[0] % 60 == 1:
                n_bright = np.sum(active_brightness > 0.01)
                max_b = active_brightness.max() if self.n_active > 0 else 0
                print(f"[frame {_frame_count[0]}] t={player.current_time_ms:.1f}ms "
                      f"bright_neurons={n_bright} max_brightness={max_b:.3f}")

            self.update_activity(active_brightness)
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
            elif key in ('q', 'Q'):
                show_manager.exit()

        # Register callbacks
        show_manager.add_timer_callback(True, 16, timer_callback)
        show_manager.window.GetInteractor().AddObserver(
            'KeyPressEvent', key_callback
        )

        print("[visualizer] Window opened. Controls:")
        print("  Mouse:  Left=Rotate, Right=Zoom, Middle=Pan")
        print("  Space:  Pause/Resume")
        print("  R:      Reset playback")
        print("  +/-:    Speed up/down")
        print("  Q:      Quit")

        show_manager.start()
