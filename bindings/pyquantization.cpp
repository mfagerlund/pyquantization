// Copyright (C) 2024, Mattias Fagerlund (pybind11 wrapper)
// Original C++ code: Copyright (C) 2022, Coudert--Osmont Yoann
// SPDX-License-Identifier: AGPL-3.0-or-later
// See <https://www.gnu.org/licenses/>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "decimation.h"
#include "quantization.h"
#include "refine/imprint.h"
#include "refine/reembed.h"

#include <sstream>
#include <unordered_set>

namespace py = pybind11;

template<> struct std::hash<std::pair<int, int>> {
	size_t operator()(const std::pair<int, int> &p) const {
		return (size_t(p.first) << (8*(sizeof(size_t) - sizeof(int)))) ^ p.second;
	}
};

// Build Mesh + seamless UVs + feature flags from numpy arrays
// This replaces readObj() from mesh.cpp
static std::tuple<Mesh, std::vector<vec2>, std::vector<bool>>
build_mesh_from_arrays(
    py::array_t<double> vertices,
    py::array_t<int32_t> triangles,
    py::array_t<double> uv_per_corner,
    py::array_t<int32_t> uv_triangles,
    py::array_t<int32_t> feature_edges
) {
    auto verts = vertices.unchecked<2>();
    auto tris = triangles.unchecked<2>();
    auto uvs = uv_per_corner.unchecked<2>();
    auto uv_tris = uv_triangles.unchecked<2>();
    auto feats = feature_edges.unchecked<2>();

    int n_verts = (int)verts.shape(0);
    int n_tris = (int)tris.shape(0);
    int n_feat = (int)feats.shape(0);

    Mesh m;
    m.points.resize(n_verts);
    for (int i = 0; i < n_verts; ++i) {
        m.points[i] = vec3(verts(i, 0), verts(i, 1), verts(i, 2));
    }

    // h2v: half-edge to vertex, 3 entries per face
    m.h2v.resize(3 * n_tris);
    std::vector<vec2> seamless(3 * n_tris);
    for (int f = 0; f < n_tris; ++f) {
        for (int j = 0; j < 3; ++j) {
            m.h2v[3*f + j] = tris(f, j);
            int uv_idx = uv_tris(f, j);
            seamless[3*f + j] = vec2(uvs(uv_idx, 0), uvs(uv_idx, 1));
        }
    }

    // Build feature edge set
    std::unordered_set<std::pair<int,int>> feat_set;
    for (int i = 0; i < n_feat; ++i) {
        int a = feats(i, 0), b = feats(i, 1);
        feat_set.emplace(std::min(a, b), std::max(a, b));
    }

    // Compute connectivity
    m.compute_opp();

    // Mark features: from explicit feature edges + boundary half-edges
    std::vector<bool> feature(m.ncorners(), false);
    for (int h = 0; h < m.ncorners(); ++h) {
        int a = m.from(h), b = m.to(h);
        feature[h] = feat_set.count({std::min(a, b), std::max(a, b)}) > 0;
    }
    for (int h = 0; h < m.ncorners(); ++h) {
        if (m.opp(h) == -1) feature[h] = true;
    }

    return {std::move(m), std::move(seamless), std::move(feature)};
}

// Convert Mesh + UVs back to numpy arrays
// This replaces writeObj() from mesh.cpp
static py::tuple mesh_to_arrays(const Mesh &m, const std::vector<vec2> &uv, const std::vector<bool> &feature) {
    int n_verts = m.nverts();
    int n_faces = m.nfacets();
    int n_corners = m.ncorners();

    // Vertices
    py::array_t<double> out_verts({n_verts, 3});
    auto v = out_verts.mutable_unchecked<2>();
    for (int i = 0; i < n_verts; ++i) {
        v(i, 0) = m.points[i].x;
        v(i, 1) = m.points[i].y;
        v(i, 2) = m.points[i].z;
    }

    // Faces
    py::array_t<int32_t> out_faces({n_faces, 3});
    auto f = out_faces.mutable_unchecked<2>();
    for (int fi = 0; fi < n_faces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            f(fi, j) = m.v(fi, j);
        }
    }

    // UVs (per-corner, flattened to unique list + index array)
    py::array_t<double> out_uvs({n_corners, 2});
    auto u = out_uvs.mutable_unchecked<2>();
    for (int h = 0; h < n_corners; ++h) {
        u(h, 0) = uv[h].x;
        u(h, 1) = uv[h].y;
    }

    // UV triangle indices (corner h maps to uv index h)
    py::array_t<int32_t> out_uv_tris({n_faces, 3});
    auto ut = out_uv_tris.mutable_unchecked<2>();
    for (int fi = 0; fi < n_faces; ++fi) {
        for (int j = 0; j < 3; ++j) {
            ut(fi, j) = 3*fi + j;
        }
    }

    // Feature edges.
    // Only meshes that carry a per-corner feature flag AND computed adjacency can
    // report feature edges. The imprinted mesh has neither (transfert_coarse_to_fine
    // builds it corner-soup style and never calls compute_opp), and indexing an
    // empty std::vector<bool> / adjacency array here is what used to segfault.
    std::vector<std::pair<int,int>> feat_list;
    if ((int)feature.size() == n_corners) {
        for (int h = 0; h < n_corners; ++h) {
            if (feature[h] && m.opp(h) < h) {
                feat_list.emplace_back(m.from(h), m.to(h));
            }
        }
    }
    py::array_t<int32_t> out_feats({(int)feat_list.size(), 2});
    auto fe = out_feats.mutable_unchecked<2>();
    for (int i = 0; i < (int)feat_list.size(); ++i) {
        fe(i, 0) = feat_list[i].first;
        fe(i, 1) = feat_list[i].second;
    }

    return py::make_tuple(out_verts, out_faces, out_uvs, out_uv_tris, out_feats);
}

enum class OutputMode {
    DECIMATED,
    IMPRINT,
    REEMBED
};

static py::tuple quantize_mesh(
    py::array_t<double> vertices,
    py::array_t<int32_t> triangles,
    py::array_t<double> uv_per_corner,
    py::array_t<int32_t> uv_triangles,
    py::array_t<int32_t> feature_edges,
    double scale,
    double scale_auto,
    const std::string &mode
) {
    // Parse mode
    OutputMode output_mode;
    if (mode == "decimate" || mode == "decimated") output_mode = OutputMode::DECIMATED;
    else if (mode == "imprint") output_mode = OutputMode::IMPRINT;
    else if (mode == "reembed") output_mode = OutputMode::REEMBED;
    else throw std::invalid_argument("mode must be 'decimate', 'imprint', or 'reembed'");

    // Build mesh from numpy arrays (replaces readObj)
    auto [m, seamless, feature] = build_mesh_from_arrays(
        vertices, triangles, uv_per_corner, uv_triangles, feature_edges
    );

    std::cerr << "INPUT: \t" << m.nverts() << " verts \t" << m.nfacets() << " facets" << std::endl;

    // Apply scale
    if (scale < 0) {
        // Auto scale
        double surface_area = 0.;
        for (const int f : m.facets()) surface_area += uvArea(m, seamless, f);
        scale = .3 * std::sqrt(m.nfacets() / surface_area);
        scale *= scale_auto;
        std::cerr << "AUTOMATIC RESCALE: " << scale << std::endl;
    }
    for (vec2 &v : seamless) v *= scale;

    // Validate (positive det)
    for (const int f : m.facets()) {
        double det = (seamless[m.h(f, 1)].x - seamless[m.h(f, 0)].x)*(seamless[m.h(f, 2)].y - seamless[m.h(f, 0)].y)
                   - (seamless[m.h(f, 1)].y - seamless[m.h(f, 0)].y)*(seamless[m.h(f, 2)].x - seamless[m.h(f, 0)].x);
        if (det <= 0.) {
            throw std::runtime_error(
                "Input is not valid! Facet " + std::to_string(f) + " has negative det."
            );
        }
    }

    // Compute cut graph
    CutGraph cg = compute_cut_graph(m, seamless);

    // Save fine mesh copy
    const Mesh fine = m;
    const std::vector<vec2> f_uv = seamless;
    const std::vector<bool> f_feature = feature;
    const CutGraph f_cg = cg;

    // Decimation
    SparseMatrix D;
    if (output_mode == OutputMode::REEMBED) D.setIdentity(2*m.ncorners());
    do { makeDelaunay(m, cg, seamless, feature, D); } while(collapse(m, cg, seamless, feature, D));
    std::cerr << "DECIMATION: \t" << m.nverts() << " verts \t" << m.nfacets() << " facets" << std::endl;

    // Quantization
    std::cerr << "QUANTIZATION" << std::endl;
    const std::vector<vec2i> quv0 = quantize(m, cg, seamless, feature, false);
    const std::vector<vec2> quv(quv0.begin(), quv0.end());

    // Output dispatch
    switch (output_mode) {
    case OutputMode::DECIMATED:
        return mesh_to_arrays(m, quv, feature);

    case OutputMode::IMPRINT: {
        std::cerr << "IMPRINTING..." << std::endl;
        const auto [fineQ, uvQ] = transfert_coarse_to_fine(fine, f_uv, m, seamless, quv);

        // transfert_coarse_to_fine reports per-facet failures on stderr and just
        // skips them, so a total failure shows up only as an empty or inconsistent
        // mesh. Say so instead of handing back a degenerate result.
        if (fineQ.nfacets() == 0 || fineQ.nverts() == 0) {
            throw std::invalid_argument(
                "Imprinting failed: no coarse facet could be imprinted onto the input mesh. "
                "Imprint needs the decimated mesh to overlap the fine mesh in UV space; "
                "this typically fails on flat/degenerate inputs where decimation collapses "
                "the mesh to a couple of facets. Use mode='reembed' or mode='decimate' instead."
            );
        }
        if ((int)uvQ.size() != fineQ.ncorners()) {
            throw std::runtime_error(
                "Imprinting produced an inconsistent mesh (" + std::to_string(uvQ.size())
                + " UVs for " + std::to_string(fineQ.ncorners()) + " corners)."
            );
        }
        for (int h = 0; h < fineQ.ncorners(); ++h) {
            const int v = fineQ.h2v[h];
            if (v < 0 || v >= fineQ.nverts()) {
                throw std::runtime_error(
                    "Imprinting produced an out-of-range vertex index (" + std::to_string(v)
                    + " of " + std::to_string(fineQ.nverts()) + ") at corner " + std::to_string(h) + "."
                );
            }
        }

        // The imprinted mesh carries no feature flags and no half-edge adjacency,
        // so no feature edges are reported for this mode.
        const std::vector<bool> no_feature;
        return mesh_to_arrays(fineQ, uvQ, no_feature);
    }

    case OutputMode::REEMBED: {
        std::cerr << "RE-EMBEDING..." << std::endl;
        const std::vector<vec2> U = reembed(fine, f_uv, f_cg, f_feature, m, quv, cg, feature, D);
        return mesh_to_arrays(fine, U, f_feature);
    }
    }

    // Unreachable
    throw std::logic_error("unreachable");
}

PYBIND11_MODULE(pyquantization, mod) {
    mod.doc() = R"doc(
        pyquantization - Quad mesh quantization without a T-mesh.

        Python bindings for the quantization algorithm by Coudert-Osmont et al. (2024).
        Takes a seamless UV parameterization on a triangle mesh and produces
        an integer-grid-aligned (quantized) parameterization.

        Reference:
            Coudert-Osmont, Y., Desobry, D., Livesu, M., Sorkine-Hornung, O.,
            Bommes, D., Lévy, B., Ray, N. (2024).
            "Quad Mesh Quantization Without a T-Mesh."
            Computer Graphics Forum.
    )doc";

    mod.def("quantize_mesh", &quantize_mesh,
        py::arg("vertices"),
        py::arg("triangles"),
        py::arg("uv_per_corner"),
        py::arg("uv_triangles"),
        py::arg("feature_edges"),
        py::arg("scale") = -1.0,
        py::arg("scale_auto") = 1.0,
        py::arg("mode") = "reembed",
        R"doc(
        Quantize a seamless UV parameterization to integer grid.

        Takes a triangle mesh with a seamless UV map and produces a quantized
        (integer-aligned) version suitable for quad mesh extraction.

        Parameters
        ----------
        vertices : ndarray, shape (n_verts, 3), float64
            3D vertex positions.
        triangles : ndarray, shape (n_tris, 3), int32
            Triangle face indices (0-based).
        uv_per_corner : ndarray, shape (n_uvs, 2), float64
            UV coordinates (texture vertices).
        uv_triangles : ndarray, shape (n_tris, 3), int32
            UV index per face corner (0-based).
        feature_edges : ndarray, shape (n_feat, 2), int32
            Hard edge vertex pairs (0-based).
        scale : float
            Scale factor for UV coordinates. Use -1.0 for automatic scaling.
        scale_auto : float
            Multiplier for automatic scale (only used when scale=-1.0).
        mode : str
            Output mode:
            - "reembed": Re-embed quantized UVs on the original mesh (recommended).
            - "imprint": Imprint quantized UVs on the original mesh.
            - "decimate": Return the decimated mesh with quantized UVs.

        Returns
        -------
        out_vertices : ndarray, shape (n_out_verts, 3), float64
            Output vertex positions.
        out_faces : ndarray, shape (n_out_faces, 3), int32
            Output triangle face indices (0-based).
        out_uvs : ndarray, shape (n_out_corners, 2), float64
            Output UV coordinates (per half-edge corner).
        out_uv_triangles : ndarray, shape (n_out_faces, 3), int32
            Output UV index per face corner (0-based).
        out_feature_edges : ndarray, shape (n_out_feat, 2), int32
            Output feature edge vertex pairs (0-based).
        )doc"
    );
}
