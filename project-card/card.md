---
oneliner: Python bindings for quad mesh quantization, turning a UV parameterization into an integer-grid-aligned one for quad extraction
tags: [mesh-processing, geometry-processing, quad-mesh, quantization, uv-parameterization, pybind11, python, cpp, research-paper]
stack: [C++, pybind11, scikit-build-core, Python]
generated: 2026-09-06
commit: b3e7750
placeholder: true
---
Wraps the C++ reference implementation of "Quad Mesh Quantization Without a T-Mesh" (Coudert-Osmont et al., 2024) as a pip-installable Python package. Takes a seamless UV parameterization on a triangle mesh and produces a quantized, integer-grid-aligned one, with reembed/imprint/decimate output modes. Working alpha with a 21-test pytest suite covering all three modes.
