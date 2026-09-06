---
oneliner: Python bindings for quad mesh quantization: snaps a seamless UV map to the integer grid for quad extraction
tags: [mesh-processing, quad-mesh, quantization, uv-parameterization, integer-grid, pybind11, cpp, python, torus, research-paper]
stack: [C++, pybind11, scikit-build-core, Python]
generated: 2026-09-06
commit: d5761dc
placeholder: false
---
Wraps the C++ reference implementation of "Quad Mesh Quantization Without a T-Mesh" (Coudert-Osmont et al., 2024) as a pip-installable Python package. Takes a seamless UV parameterization on a triangle mesh and produces a quantized, integer-grid-aligned one, with reembed/imprint/decimate output modes. Working alpha with a 21-test pytest suite; decimate is the robust mode, while reembed sometimes diverges to NaN and imprint can crash on degenerate synthetic input.
