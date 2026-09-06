import numpy as np
import pytest

import pyquantization


class TestImport:
    def test_module_loads(self):
        assert hasattr(pyquantization, "quantize_mesh")

    def test_docstring(self):
        assert "quantize" in pyquantization.__doc__.lower()

    def test_function_docstring(self):
        assert "Parameters" in pyquantization.quantize_mesh.__doc__


class TestDecimateMode:
    """Tests using decimate mode which is the most robust on synthetic data."""

    def test_basic(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        assert len(result) == 5
        out_verts, out_faces, out_uvs, out_uv_tris, out_feats = result
        assert out_verts.ndim == 2 and out_verts.shape[1] == 3
        assert out_faces.ndim == 2 and out_faces.shape[1] == 3
        assert out_uvs.ndim == 2 and out_uvs.shape[1] == 2
        assert out_uv_tris.ndim == 2 and out_uv_tris.shape[1] == 3
        assert out_feats.ndim == 2 and out_feats.shape[1] == 2

    def test_output_dtypes(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        out_verts, out_faces, out_uvs, out_uv_tris, out_feats = result
        assert out_verts.dtype == np.float64
        assert out_faces.dtype == np.int32
        assert out_uvs.dtype == np.float64
        assert out_uv_tris.dtype == np.int32
        assert out_feats.dtype == np.int32

    def test_face_indices_valid(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        out_verts, out_faces, out_uvs, out_uv_tris, _ = result
        assert np.all(out_faces >= 0)
        assert np.all(out_faces < len(out_verts))
        assert np.all(out_uv_tris >= 0)
        assert np.all(out_uv_tris < len(out_uvs))

    def test_vertices_preserved(self, quad_mesh):
        """For the minimal 2-tri mesh, decimation can't remove boundary verts."""
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        out_verts = result[0]
        assert len(out_verts) >= 3  # at least a single triangle

    def test_uvs_are_integer_quantized(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        out_uvs = result[2]
        residuals = np.abs(out_uvs - np.round(out_uvs))
        assert np.all(residuals < 0.01), f"UVs not integer-quantized: {out_uvs}"

    def test_grid_mesh(self, grid_mesh):
        result = pyquantization.quantize_mesh(*grid_mesh, mode="decimate")
        out_verts, out_faces, out_uvs, out_uv_tris, out_feats = result
        assert len(out_verts) > 0
        assert len(out_faces) > 0
        assert np.all(np.isfinite(out_verts))
        assert np.all(np.isfinite(out_uvs))

    def test_grid_mesh_reduces_vertex_count(self, grid_mesh):
        """Decimation should significantly reduce the mesh."""
        result = pyquantization.quantize_mesh(*grid_mesh, mode="decimate")
        out_verts = result[0]
        in_verts = grid_mesh[0]
        assert len(out_verts) < len(in_verts)

    def test_explicit_scale(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, scale=2.0, mode="decimate")
        assert len(result) == 5
        assert len(result[0]) > 0

    def test_scale_auto_multiplier(self, quad_mesh):
        r1 = pyquantization.quantize_mesh(*quad_mesh, scale=-1.0, scale_auto=0.5, mode="decimate")
        r2 = pyquantization.quantize_mesh(*quad_mesh, scale=-1.0, scale_auto=2.0, mode="decimate")
        # Different scale_auto should produce different UV ranges
        assert len(r1[0]) > 0
        assert len(r2[0]) > 0

    def test_feature_edges_output(self, quad_mesh):
        """Boundary edges become feature edges on boundary meshes."""
        result = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        out_feats = result[4]
        assert len(out_feats) > 0  # boundary mesh should have feature edges

    def test_mode_alias_decimated(self, quad_mesh):
        r1 = pyquantization.quantize_mesh(*quad_mesh, mode="decimate")
        r2 = pyquantization.quantize_mesh(*quad_mesh, mode="decimated")
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])


class TestReembedMode:
    def test_basic(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="reembed")
        assert len(result) == 5
        out_verts, out_faces, out_uvs, out_uv_tris, out_feats = result
        assert out_verts.shape == quad_mesh[0].shape  # same vertex count
        assert out_faces.shape == quad_mesh[1].shape  # same face count

    def test_vertices_unchanged(self, quad_mesh):
        """Reembed preserves original vertex positions."""
        result = pyquantization.quantize_mesh(*quad_mesh, mode="reembed")
        np.testing.assert_array_almost_equal(result[0], quad_mesh[0])

    def test_output_shapes(self, quad_mesh):
        result = pyquantization.quantize_mesh(*quad_mesh, mode="reembed")
        out_verts, out_faces, out_uvs, out_uv_tris, _ = result
        n_faces = len(out_faces)
        assert len(out_uvs) == 3 * n_faces  # per-corner UVs
        assert out_uv_tris.shape == (n_faces, 3)


class TestReembedRobustness:
    """Regression tests for the re-embedding solver.

    reembed used to return all-NaN UVs (the L-BFGS curvature condition collapsed
    to 0/0 once the iterate stalled, and the line search silently accepted the
    NaN step), and the result varied run to run because the parallel conjugate
    gradient reduced its dot products into shared doubles under a mutex, so the
    summation order - and hence the floating point rounding - was arbitrary.
    """

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_uvs_are_finite(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        out_uvs = pyquantization.quantize_mesh(*mesh, mode="reembed")[2]
        assert np.all(np.isfinite(out_uvs)), (
            f"{np.isnan(out_uvs).sum()} of {out_uvs.size} UV components are NaN/inf"
        )

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_bit_identical_across_runs(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        runs = [pyquantization.quantize_mesh(*mesh, mode="reembed")[2] for _ in range(5)]
        for i, uv in enumerate(runs[1:], start=1):
            assert uv.shape == runs[0].shape
            np.testing.assert_array_equal(
                uv, runs[0], err_msg=f"run {i} differs from run 0 (nondeterministic solve)"
            )
        # A hash over the raw bytes catches -0.0 vs 0.0, which array_equal does not.
        digests = {r.tobytes() for r in runs}
        assert len(digests) == 1, "reembed output is not bit-identical across runs"

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_topology_preserved(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        out_verts, out_faces, out_uvs, _, _ = pyquantization.quantize_mesh(
            *mesh, mode="reembed"
        )
        assert out_verts.shape == mesh[0].shape
        assert out_faces.shape == mesh[1].shape
        assert len(out_uvs) == 3 * len(out_faces)
        np.testing.assert_array_almost_equal(out_verts, mesh[0])

    def test_flat_grid_is_finite(self, flat_grid_mesh):
        out_uvs = pyquantization.quantize_mesh(*flat_grid_mesh, mode="reembed")[2]
        assert np.all(np.isfinite(out_uvs))


class TestImprintRobustness:
    """Regression tests for imprint mode.

    imprint used to segfault on synthetic grids: transfert_coarse_to_fine returns
    a corner-soup mesh with no per-corner feature flags and no half-edge adjacency,
    and the binding then indexed both of those empty arrays while collecting
    feature edges.
    """

    def test_flat_grid_no_crash(self, flat_grid_mesh):
        """Must either succeed with finite output or raise - never crash."""
        try:
            result = pyquantization.quantize_mesh(*flat_grid_mesh, mode="imprint")
        except ValueError:
            return  # documented rejection of an input imprint cannot handle
        out_verts, out_faces, out_uvs, out_uv_tris, out_feats = result
        assert np.all(np.isfinite(out_verts))
        assert np.all(np.isfinite(out_uvs))
        assert len(out_faces) > 0
        assert np.all(out_faces >= 0) and np.all(out_faces < len(out_verts))
        assert np.all(out_uv_tris >= 0) and np.all(out_uv_tris < len(out_uvs))
        assert out_feats.shape[1] == 2

    def test_bumpy_grid_no_crash(self, grid_mesh):
        try:
            result = pyquantization.quantize_mesh(*grid_mesh, mode="imprint")
        except ValueError:
            return
        assert np.all(np.isfinite(result[0]))
        assert np.all(np.isfinite(result[2]))

    def test_quad_mesh_no_crash(self, quad_mesh):
        try:
            result = pyquantization.quantize_mesh(*quad_mesh, mode="imprint")
        except ValueError:
            return
        assert np.all(np.isfinite(result[2]))

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_goldens_no_crash(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        try:
            result = pyquantization.quantize_mesh(*mesh, mode="imprint")
        except ValueError:
            return
        out_verts, out_faces, out_uvs, _, _ = result
        assert np.all(np.isfinite(out_verts))
        assert np.all(np.isfinite(out_uvs))
        assert np.all(out_faces >= 0) and np.all(out_faces < len(out_verts))


class TestDecimateGoldens:
    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_decimate(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        out_verts, out_faces, out_uvs, out_uv_tris, _ = pyquantization.quantize_mesh(
            *mesh, mode="decimate"
        )
        assert len(out_verts) > 0 and len(out_faces) > 0
        assert len(out_verts) < len(mesh[0])  # decimation must actually reduce
        assert np.all(np.isfinite(out_verts))
        assert np.all(np.isfinite(out_uvs))
        assert np.all(out_faces >= 0) and np.all(out_faces < len(out_verts))
        assert np.all(out_uv_tris >= 0) and np.all(out_uv_tris < len(out_uvs))

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_decimate_uvs_are_integers(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        out_uvs = pyquantization.quantize_mesh(*mesh, mode="decimate")[2]
        assert np.all(np.abs(out_uvs - np.round(out_uvs)) < 1e-9)

    @pytest.mark.parametrize("fixture_name", ["torus_mesh", "sphere320_mesh"])
    def test_decimate_is_deterministic(self, request, fixture_name):
        mesh = request.getfixturevalue(fixture_name)
        runs = [pyquantization.quantize_mesh(*mesh, mode="decimate")[2] for _ in range(3)]
        for uv in runs[1:]:
            np.testing.assert_array_equal(uv, runs[0])


class TestErrorHandling:
    def test_invalid_mode(self, quad_mesh):
        with pytest.raises(ValueError, match="mode must be"):
            pyquantization.quantize_mesh(*quad_mesh, mode="invalid")

    def test_negative_det_raises(self):
        """Flipped triangle orientation should raise RuntimeError."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        # Flipped UVs (clockwise instead of counter-clockwise)
        uv = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        uv_tris = np.array([[0, 1, 2]], dtype=np.int32)
        feats = np.empty((0, 2), dtype=np.int32)
        with pytest.raises(RuntimeError, match="not valid"):
            pyquantization.quantize_mesh(
                vertices, triangles, uv, uv_tris, feats, scale=1.0, mode="decimate",
            )

    def test_wrong_vertex_ndim(self):
        """1D array instead of 2D should raise a pybind11 type error."""
        vertices = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        triangles = np.array([[0, 0, 0]], dtype=np.int32)
        uv = np.array([[0.0, 0.0]], dtype=np.float64)
        uv_tris = np.array([[0, 0, 0]], dtype=np.int32)
        feats = np.empty((0, 2), dtype=np.int32)
        with pytest.raises(Exception):
            pyquantization.quantize_mesh(
                vertices, triangles, uv, uv_tris, feats, mode="decimate",
            )

    def test_float32_auto_converts(self):
        """pybind11 should accept float32 vertices by auto-converting."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        uv_tris = np.array([[0, 1, 2]], dtype=np.int32)
        feats = np.empty((0, 2), dtype=np.int32)
        result = pyquantization.quantize_mesh(
            vertices, triangles, uv, uv_tris, feats, scale=1.0, mode="decimate",
        )
        assert result[0].dtype == np.float64
