import math
import numpy as np

from fury import window

try:
    import vtk
    from vtk.util import numpy_support
except Exception as e:
    raise RuntimeError(
        "This script needs VTK (normally installed together with FURY)."
    ) from e


# ============================================================
# FURY / VTK neuron flash playground v9
#
# Main fix vs v8:
#   - base points are suppressed by BOTH self activity and a local-neighborhood
#     hot field, so neurons sitting inside a nearby avalanche do not leave tiny
#     gray dots inside the white bloom.
#
# Design:
#   - one tiny warm base-point actor for low-activity / low-neighborhood-hot neurons
#   - four gaussian splat actors for aura / bloom / core / hot-white
#   - base points are culled when either self OR nearby-neuron intensity is high
# ============================================================

# --------------------------- global scene ----------------------------
N = 3800
SEED = 12
FPS = 60
DT = 1.0 / FPS
BACKGROUND = (0.0, 0.0, 0.0)
WINDOW_SIZE = (1400, 900)

# ------------------------- temporal dynamics -------------------------
BASE_RATE_HZ = 0.20
GROUP_BURST_RATE_HZ = 6.8
GLOBAL_BURST_CHANCE = 0.018
GROUP_BURST_CHANCE = 0.042

RISE_MS = 6.0
DECAY_MS = 108.0
SATURATION = 3.3
PEAK_SHOULDER = 2.25

# ---------------------------- look tuning ----------------------------
BASE_POINT_SIZE = 2.7

AURA_SCALE_MULT = 0.72
BLOOM_SCALE_MULT = 1.34
CORE_SCALE_MULT = 0.35
HOT_SCALE_MULT = 0.22

AURA_CENTER_RELIEF = 0.08

WHITE_KNEE = 0.89
BLOOM_KNEE = 0.685

AURA_GAIN = 1.10
BLOOM_GAIN = 1.42
CORE_GAIN = 1.42
HOT_GAIN = 2.35

# Base-point suppression controls.
# The key fix in v9 is that base points are hidden not only by their own activity,
# but also by the LOCAL glow field from nearby hot neurons.
BASE_HIDE_START = 0.26   # base starts fading here
BASE_HIDE_END = 0.42     # base should be effectively gone by here
BASE_HARD_HIDE = 0.48    # absolutely remove from base actor above this
BASE_KEEP_EPS = 0.012    # below this visibility, point is not included at all

LOCAL_HIDE_START = 0.20  # nearby hot neurons start suppressing base here
LOCAL_HIDE_END = 0.40    # by here, local suppression is strong
LOCAL_HARD_HIDE = 0.36   # absolutely remove base points inside strong local avalanches
LOCAL_HIDE_POWER = 1.75  # >1 = stronger neighborhood suppression near peaks
LOCAL_SIGMA_MULT = 1.05  # gaussian falloff radius relative to bloom scale
LOCAL_RADIUS_MULT = 2.4  # neighborhood cutoff radius in sigmas
LOCAL_WEIGHT_MIN = 0.045 # ignore tiny neighbor contributions

# ----------------------------- point cloud ---------------------------
rng = np.random.default_rng(SEED)

n_groups = 10
group_ids = rng.integers(0, n_groups, size=N)
cluster_centers = np.column_stack(
    [
        rng.normal(0.0, 58.0, n_groups),
        rng.normal(0.0, 26.0, n_groups),
        rng.normal(0.0, 18.0, n_groups),
    ]
).astype(np.float32)
cluster_scales = np.column_stack(
    [
        rng.uniform(7.0, 17.0, n_groups),
        rng.uniform(5.0, 11.0, n_groups),
        rng.uniform(4.0, 9.0, n_groups),
    ]
).astype(np.float32)

points = (
    cluster_centers[group_ids]
    + rng.normal(size=(N, 3)).astype(np.float32) * cluster_scales[group_ids]
)
points[:, 0] += 0.18 * points[:, 1]
points[:, 2] += 0.07 * points[:, 0]
points = np.ascontiguousarray(points.astype(np.float32))

bbox_min = points.min(axis=0)
bbox_max = points.max(axis=0)
diag = float(np.linalg.norm(bbox_max - bbox_min))
world_unit = diag / 125.0

AURA_SCALE = AURA_SCALE_MULT * world_unit
BLOOM_SCALE = BLOOM_SCALE_MULT * world_unit
CORE_SCALE = CORE_SCALE_MULT * world_unit
HOT_SCALE = HOT_SCALE_MULT * world_unit

LOCAL_SIGMA = max(LOCAL_SIGMA_MULT * BLOOM_SCALE, 0.30 * world_unit)
LOCAL_RADIUS = LOCAL_RADIUS_MULT * LOCAL_SIGMA


# ---------------------------- math helpers ---------------------------
def maybe_call(obj, method_name, *args):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        return fn(*args)
    return None



def smoothstep(edge0, edge1, x):
    denom = max(edge1 - edge0, 1e-8)
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_local_suppression_graph(pts, radius, sigma, weight_min=0.045):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    n = len(pts)
    neigh_idx = []
    neigh_w = []
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        all_inds = tree.query_ball_point(pts, r=float(radius))
        for i, inds in enumerate(all_inds):
            inds = [j for j in inds if j != i]
            if not inds:
                neigh_idx.append(np.empty(0, dtype=np.int32))
                neigh_w.append(np.empty(0, dtype=np.float32))
                continue
            ij = np.asarray(inds, dtype=np.int32)
            d2 = np.sum((pts[ij] - pts[i]) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / max(sigma * sigma, 1e-8)).astype(np.float32)
            keep = w >= float(weight_min)
            neigh_idx.append(ij[keep])
            neigh_w.append(w[keep])
        return neigh_idx, neigh_w
    except Exception:
        pass

    r2 = float(radius * radius)
    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = pts[i0:i1]
        d = block[:, None, :] - pts[None, :, :]
        d2 = np.sum(d * d, axis=2)
        for bi in range(i1 - i0):
            i = i0 + bi
            mask = d2[bi] <= r2
            mask[i] = False
            inds = np.nonzero(mask)[0].astype(np.int32)
            if inds.size == 0:
                neigh_idx.append(np.empty(0, dtype=np.int32))
                neigh_w.append(np.empty(0, dtype=np.float32))
                continue
            w = np.exp(-0.5 * d2[bi, inds] / max(sigma * sigma, 1e-8)).astype(np.float32)
            keep = w >= float(weight_min)
            neigh_idx.append(inds[keep])
            neigh_w.append(w[keep])
    return neigh_idx, neigh_w


def compute_local_hot_field(activity_vals, neigh_idx, neigh_w, power=1.55):
    a = np.clip(activity_vals.astype(np.float32), 0.0, 1.0)
    field = a.copy()
    for i in range(len(a)):
        idx = neigh_idx[i]
        if idx.size == 0:
            continue
        local = np.max(neigh_w[i] * a[idx])
        field[i] = max(field[i], float(local))
    if power != 1.0:
        field = np.power(np.clip(field, 0.0, 1.0), power)
    return field.astype(np.float32)


LOCAL_NEIGH_IDX, LOCAL_NEIGH_W = build_local_suppression_graph(
    points,
    radius=LOCAL_RADIUS,
    sigma=LOCAL_SIGMA,
    weight_min=LOCAL_WEIGHT_MIN,
)


# --------------------------- color constants -------------------------
BLACK = (0.0, 0.0, 0.0)
DEEP_RED = (0.08, 0.00, 0.00)
BLOOD_RED = (0.30, 0.01, 0.00)
ORANGE_RED = (0.92, 0.07, 0.01)
AMBER = (1.00, 0.43, 0.02)
YELLOW = (1.00, 0.84, 0.12)
PALE_YELLOW = (1.00, 0.93, 0.62)
WARM_WHITE = (1.00, 0.992, 0.972)

# Make low-activity bodies warm and ember-like, not gray.
BASE_OFF_0 = np.array([0.070, 0.020, 0.006], dtype=np.float32)
BASE_OFF_1 = np.array([0.220, 0.078, 0.016], dtype=np.float32)
base_mix = rng.uniform(0.0, 1.0, size=(N, 1)).astype(np.float32)
base_rgb_off = BASE_OFF_0[None, :] * (1.0 - base_mix) + BASE_OFF_1[None, :] * base_mix
base_alpha_off = (0.18 + 0.08 * base_mix[:, 0]).astype(np.float32)


# -------------------- polydata / vtk data helpers --------------------
def make_verts(n_points):
    verts = vtk.vtkCellArray()
    for i in range(int(n_points)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    return verts


def set_point_polydata_with_colors(poly, pts, rgba):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(pts, deep=True))

    colors = numpy_support.numpy_to_vtk(
        rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    colors.SetNumberOfComponents(4)
    colors.SetName("colors")

    poly.SetPoints(vtk_points)
    poly.SetVerts(make_verts(len(pts)))
    poly.GetPointData().SetScalars(colors)
    poly.Modified()



def make_empty_point_polydata():
    poly = vtk.vtkPolyData()
    empty_pts = np.empty((0, 3), dtype=np.float32)
    empty_rgba = np.empty((0, 4), dtype=np.uint8)
    set_point_polydata_with_colors(poly, empty_pts, empty_rgba)
    return poly



def make_splat_polydata(pts, activity):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(pts, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(make_verts(len(pts)))

    activity_vtk = numpy_support.numpy_to_vtk(activity, deep=False)
    activity_vtk.SetName("activity")
    poly.GetPointData().AddArray(activity_vtk)
    poly.GetPointData().SetActiveScalars("activity")
    return poly, activity_vtk



def make_color_tf(stops):
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToRGB()
    for x, (r, g, b) in stops:
        ctf.AddRGBPoint(float(x), float(r), float(g), float(b))
    return ctf



def make_opacity_tf(stops, gain=1.0):
    otf = vtk.vtkPiecewiseFunction()
    for x, a in stops:
        otf.AddPoint(float(x), float(np.clip(a * gain, 0.0, 1.0)))
    return otf



def make_base_point_actor(poly, point_size):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOn()
    mapper.SetColorModeToDirectScalars()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetRepresentationToPoints()
    prop.SetPointSize(point_size)
    prop.LightingOff()
    try:
        prop.SetRenderPointsAsSpheres(False)
    except Exception:
        pass
    return actor



def splat_shader(kind):
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



def make_gaussian_actor(poly, kind, scale_factor, color_tf, opacity_tf):
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetInputData(poly)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("activity")
    maybe_call(mapper, "SetOpacityArray", "activity")
    maybe_call(mapper, "SetOpacityArrayComponent", 0)
    maybe_call(mapper, "SetScalarRange", 0.0, 1.0)
    maybe_call(mapper, "UseLookupTableScalarRangeOn")
    maybe_call(mapper, "SetLookupTable", color_tf)
    maybe_call(mapper, "SetScalarOpacityFunction", opacity_tf)
    maybe_call(mapper, "SetScaleFactor", float(scale_factor))
    maybe_call(mapper, "SetSplatShaderCode", splat_shader(kind))
    maybe_call(mapper, "SetEmissive", 1)
    maybe_call(mapper, "EmissiveOn")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.LightingOff()
    prop.SetOpacity(1.0)
    return actor


# -------------------- transfer functions per layer -------------------
aura_ctf = make_color_tf(
    [
        (0.00, BLACK),
        (0.04, DEEP_RED),
        (0.16, BLOOD_RED),
        (0.42, ORANGE_RED),
        (0.76, AMBER),
        (1.00, YELLOW),
    ]
)
aura_otf = make_opacity_tf(
    [
        (0.00, 0.00),
        (0.04, 0.00),
        (0.12, 0.10),
        (0.34, 0.38),
        (0.68, 0.78),
        (1.00, 1.00),
    ],
    gain=AURA_GAIN,
)

bloom_ctf = make_color_tf(
    [
        (0.00, BLACK),
        (BLOOM_KNEE - 0.06, BLACK),
        (BLOOM_KNEE, ORANGE_RED),
        (0.84, YELLOW),
        (0.95, PALE_YELLOW),
        (1.00, WARM_WHITE),
    ]
)
bloom_otf = make_opacity_tf(
    [
        (0.00, 0.00),
        (BLOOM_KNEE - 0.07, 0.00),
        (BLOOM_KNEE, 0.08),
        (0.82, 0.30),
        (0.92, 0.72),
        (1.00, 1.00),
    ],
    gain=BLOOM_GAIN,
)

core_ctf = make_color_tf(
    [
        (0.00, BLACK),
        (0.05, DEEP_RED),
        (0.21, BLOOD_RED),
        (0.50, ORANGE_RED),
        (0.78, YELLOW),
        (0.91, PALE_YELLOW),
        (1.00, WARM_WHITE),
    ]
)
core_otf = make_opacity_tf(
    [
        (0.00, 0.00),
        (0.05, 0.00),
        (0.14, 0.12),
        (0.42, 0.56),
        (0.78, 0.96),
        (1.00, 1.00),
    ],
    gain=CORE_GAIN,
)

hot_ctf = make_color_tf(
    [
        (0.00, BLACK),
        (WHITE_KNEE - 0.05, BLACK),
        (WHITE_KNEE, AMBER),
        (0.922, PALE_YELLOW),
        (0.948, WARM_WHITE),
        (0.974, (1.0, 1.0, 1.0)),
        (1.00, (1.0, 1.0, 1.0)),
    ]
)
hot_otf = make_opacity_tf(
    [
        (0.00, 0.00),
        (WHITE_KNEE - 0.06, 0.00),
        (WHITE_KNEE, 0.28),
        (0.922, 0.86),
        (0.948, 1.00),
        (0.974, 1.00),
        (1.00, 1.00),
    ],
    gain=HOT_GAIN,
)


# ----------------------- firing / intensity model --------------------
a = math.exp(-(DT * 1000.0) / DECAY_MS)
b = math.exp(-(DT * 1000.0) / RISE_MS)


def single_spike_peak(a_, b_, steps=400):
    slow = 1.0
    fast = 1.0
    peak = 0.0
    for _ in range(steps):
        slow *= a_
        fast *= b_
        peak = max(peak, slow - fast)
    return peak


PEAK_NORM = 1.0 / max(single_spike_peak(a, b), 1e-8)
slow_state = np.zeros(N, dtype=np.float32)
fast_state = np.zeros(N, dtype=np.float32)

group_drive = np.zeros(n_groups, dtype=np.float32)
global_drive = 0.0



def update_firing_and_intensity():
    global slow_state, fast_state, group_drive, global_drive

    if rng.random() < GROUP_BURST_CHANCE:
        g = rng.integers(0, n_groups)
        group_drive[g] += rng.uniform(0.85, 1.55)

    if rng.random() < GLOBAL_BURST_CHANCE:
        global_drive += rng.uniform(0.24, 0.66)

    group_drive *= 0.915
    global_drive *= 0.94

    rates = np.full(N, BASE_RATE_HZ, dtype=np.float32)
    rates += GROUP_BURST_RATE_HZ * group_drive[group_ids]
    rates += 1.85 * GROUP_BURST_RATE_HZ * global_drive

    spikes = (rng.random(N) < (rates * DT)).astype(np.float32)

    slow_state = slow_state * a + spikes
    fast_state = fast_state * b + spikes
    raw = np.maximum((slow_state - fast_state) * PEAK_NORM, 0.0)

    saturated = 1.0 - np.exp(-SATURATION * raw)
    shoulder = 1.0 - np.exp(-PEAK_SHOULDER * np.maximum(saturated - 0.10, 0.0))
    combined = np.clip(0.64 * saturated + 0.82 * shoulder, 0.0, 1.0)

    shaped = np.power(combined, 0.62)
    shaped = np.clip(
        shaped + 0.22 * np.power(np.clip(combined - 0.72, 0.0, 1.0), 0.45),
        0.0,
        1.0,
    )
    return shaped.astype(np.float32)


# ------------------------------ build scene --------------------------
scene = window.Scene()
scene.background(BACKGROUND)


def build_base_subset_from_activity(activity_vals):
    a = np.clip(activity_vals.astype(np.float32), 0.0, 1.0)

    # Real fix: a neuron can look 'inside a white blob' because nearby neurons are hot,
    # even if this neuron's own activity is only modest. So suppress the base point
    # using both self activity and a local neighborhood hot-field.
    local_hot = compute_local_hot_field(
        a,
        LOCAL_NEIGH_IDX,
        LOCAL_NEIGH_W,
        power=LOCAL_HIDE_POWER,
    )

    self_gate = 1.0 - smoothstep(BASE_HIDE_START, BASE_HIDE_END, a)
    local_gate = 1.0 - smoothstep(LOCAL_HIDE_START, LOCAL_HIDE_END, local_hot)
    gate = np.minimum(self_gate, local_gate)

    gate[a >= BASE_HARD_HIDE] = 0.0
    gate[local_hot >= LOCAL_HARD_HIDE] = 0.0
    keep = gate > BASE_KEEP_EPS

    if not np.any(keep):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.uint8),
        )

    vis = np.power(gate[keep], 2.4)
    heat = np.power(a[keep], 1.15)

    # Keep the base warm, never neutral gray. No visibility floor: hidden means hidden.
    ember_tint = np.array([1.0, 0.62, 0.18], dtype=np.float32)
    rgb = base_rgb_off[keep] * (0.98 * vis[:, None])
    rgb += ember_tint[None, :] * (0.08 * heat[:, None] * vis[:, None])

    alpha = base_alpha_off[keep] * vis

    rgba = np.empty((keep.sum(), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    rgba[:, 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)

    return points[keep], rgba


base_poly = make_empty_point_polydata()
base_actor = make_base_point_actor(base_poly, BASE_POINT_SIZE)

activity = np.zeros(N, dtype=np.float32)
splat_poly, activity_vtk = make_splat_polydata(points, activity)

aura_actor = make_gaussian_actor(
    splat_poly,
    kind="aura",
    scale_factor=AURA_SCALE,
    color_tf=aura_ctf,
    opacity_tf=aura_otf,
)
bloom_actor = make_gaussian_actor(
    splat_poly,
    kind="bloom",
    scale_factor=BLOOM_SCALE,
    color_tf=bloom_ctf,
    opacity_tf=bloom_otf,
)
core_actor = make_gaussian_actor(
    splat_poly,
    kind="core",
    scale_factor=CORE_SCALE,
    color_tf=core_ctf,
    opacity_tf=core_otf,
)
hot_actor = make_gaussian_actor(
    splat_poly,
    kind="hot",
    scale_factor=HOT_SCALE,
    color_tf=hot_ctf,
    opacity_tf=hot_otf,
)

scene.add(base_actor)
scene.add(aura_actor)
scene.add(bloom_actor)
scene.add(core_actor)
scene.add(hot_actor)

cam_z = diag * 1.55
scene.set_camera(
    position=(0.0, 0.0, cam_z),
    focal_point=(0.0, 0.0, 0.0),
    view_up=(0.0, 1.0, 0.0),
)

showm = window.ShowManager(scene, size=WINDOW_SIZE, reset_camera=False)


# ------------------------------ animation ----------------------------
def timer_callback(_obj, _event):
    activity[:] = update_firing_and_intensity()

    base_pts, base_rgba = build_base_subset_from_activity(activity)
    set_point_polydata_with_colors(base_poly, base_pts, base_rgba)

    activity_vtk.Modified()
    splat_poly.Modified()
    showm.render()


print("FURY neuron flash playground v10")
print(f"  points: {N}")
print(f"  world-unit: {world_unit:.4f}")
print(
    f"  aura/core/hot/bloom scales: {AURA_SCALE:.4f} / {CORE_SCALE:.4f} / {HOT_SCALE:.4f} / {BLOOM_SCALE:.4f}"
)
print(f"  white knee: {WHITE_KNEE:.3f}, bloom knee: {BLOOM_KNEE:.3f}")
print(
    f"  base hide: start={BASE_HIDE_START:.2f}, end={BASE_HIDE_END:.2f}, hard={BASE_HARD_HIDE:.2f}, eps={BASE_KEEP_EPS:.3f}"
)
print(
    f"  local hide: start={LOCAL_HIDE_START:.2f}, end={LOCAL_HIDE_END:.2f}, hard={LOCAL_HARD_HIDE:.2f}, power={LOCAL_HIDE_POWER:.2f}"
)
print("  base points are culled by self activity AND neighborhood-hot field")

showm.add_timer_callback(True, int(1000 / FPS), timer_callback)
showm.initialize()
showm.start()
