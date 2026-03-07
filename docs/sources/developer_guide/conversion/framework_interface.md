# Framework Integration

AscendNPU IR is an MLIR-based compilation framework. It supports upstream integration with languages and frameworks such as **Triton** and **TileLang**, so that third-party DSLs and frameworks can target Ascend hardware and run custom operators on the NPU.

This page gives a short overview and indexes the integration guides in this directory.

## Index

| Integration | Description | Document |
|-------------|-------------|----------|
| **Triton** | Use Triton to write high-performance kernels and run them on Ascend NPU via Triton Ascend. Covers installation, environment, op mapping, and Ascend extensions. | [Triton interface](triton_interface.md) |
| **TileLang** | Use TileLang Ascend (tile-lang/TVM-based DSL) to develop kernels for Ascend NPU (e.g. GEMM, vector ops, attention). Covers environment, build, and quick start. | [TileLang interface](tile_lang_interface.md) |

For IR-level concepts, common compile options, and other framework paths (e.g. Torch, Linalg, HIVM), see [Interface API](interface_api.md).
