import os

import numpy as np
import pytest


# Golden seamless parameterizations live in a sibling repo. They are optional:
# tests that need them skip when it is not checked out next to this one.
GOLDEN_DIR = os.environ.get(
    "PYQUANTIZATION_GOLDEN_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "rectangular-surface-parameterization", "tests", "golden_data",
    ),
)


def load_golden(name):
    """Load `<name>_param.npz` and convert it to quantize_mesh's argument tuple."""
    path = os.path.join(GOLDEN_DIR, f"{name}_param.npz")
    if not os.path.exists(path):
        pytest.skip(f"golden parameterization not available: {path}")
    d = np.load(path)
    vertices = np.ascontiguousarray(d["vertices"], dtype=np.float64)
    triangles = np.ascontiguousarray(d["triangles"], dtype=np.int32)
    n_tris = triangles.shape[0]
    # uv_per_tri is (n_tris, 3, 2): one UV per corner, already seamless.
    uv_per_corner = np.ascontiguousarray(
        d["uv_per_tri"].reshape(3 * n_tris, 2), dtype=np.float64
    )
    uv_triangles = np.arange(3 * n_tris, dtype=np.int32).reshape(n_tris, 3)
    feature_edges = np.empty((0, 2), dtype=np.int32)
    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges


@pytest.fixture
def torus_mesh():
    """Golden seamless parameterization of a torus (576 verts, 1152 tris)."""
    return load_golden("torus")


@pytest.fixture
def sphere320_mesh():
    """Golden seamless parameterization of a sphere (162 verts, 320 tris)."""
    return load_golden("sphere320")


@pytest.fixture(params=[4, 6, 10], ids=lambda n: f"{n}x{n}")
def flat_grid_mesh(request):
    """Regular n x n planar grid with a trivial seamless param (scaled xy)."""
    n = request.param
    xs = np.linspace(0.0, 3.0, n + 1)
    xx, yy = np.meshgrid(xs, xs)
    vertices = np.column_stack([
        xx.ravel(), yy.ravel(), np.zeros(xx.size),
    ]).astype(np.float64)

    triangles = []
    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v10 = v00 + 1
            v01 = (i + 1) * (n + 1) + j
            v11 = v01 + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles = np.array(triangles, dtype=np.int32)

    uv_per_corner = (vertices[:, :2] * 2.0).copy()
    uv_triangles = triangles.copy()
    feature_edges = np.empty((0, 2), dtype=np.int32)
    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges


@pytest.fixture
def quad_mesh():
    """Minimal valid mesh: two triangles forming a unit square."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv_per_corner = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)
    uv_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    feature_edges = np.empty((0, 2), dtype=np.int32)
    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges


@pytest.fixture
def grid_mesh():
    """8x8 bumpy grid mesh (128 triangles) with feature edges."""
    n = 8
    xs = np.linspace(0, 3, n + 1)
    ys = np.linspace(0, 3, n + 1)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.3 * np.sin(xx * 2) * np.cos(yy * 2)
    vertices = np.column_stack([
        xx.ravel(), yy.ravel(), zz.ravel(),
    ]).astype(np.float64)

    triangles = []
    for i in range(n):
        for j in range(n):
            v00 = i * (n + 1) + j
            v10 = i * (n + 1) + j + 1
            v01 = (i + 1) * (n + 1) + j
            v11 = (i + 1) * (n + 1) + j + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles = np.array(triangles, dtype=np.int32)

    uv_per_corner = vertices[:, :2].copy() * 2.0
    uv_triangles = triangles.copy()

    feat = []
    mid = n // 2
    for j in range(n):
        feat.append([mid * (n + 1) + j, mid * (n + 1) + j + 1])
        feat.append([j * (n + 1) + mid, (j + 1) * (n + 1) + mid])
    feature_edges = np.array(feat, dtype=np.int32)

    return vertices, triangles, uv_per_corner, uv_triangles, feature_edges
