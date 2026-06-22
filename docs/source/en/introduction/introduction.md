# Introduction

**AscendNPU IR** (AscendNPU Intermediate Representation) is an MLIR-based (Multi-Level Intermediate Representation) intermediate representation for Ascend-affinity operator compilation. It provides complete Ascend expression capability, improves Ascend AI processor efficiency through compilation optimizations, and supports flexible integration with ecosystem frameworks via open interfaces to efficiently enable Ascend AI processors.

## Key capabilities

- **Multi-level abstraction and ease of use**  
  High-level abstraction interfaces hide Ascend computation, data movement, and synchronization details. The compiler automatically adapts to hardware and maps hardware-agnostic expressions to low-level instructions. Fine-grained performance control interfaces allow precise control of on-chip memory layout, pipeline synchronization insertion points, and ping-pong pipeline enablement for both usability and performance tuning.

- **Layered dialects and compilation**  
  - **HFusion**: Linalg-based extension for hardware-agnostic optimization and ecosystem integration; conversions to/from Arith, Math, Torch, etc.; tensor simplification, type legalization, and operator fusion.  
  - **HIVM**: Tile-level abstraction for Ascend computation, data movement, and synchronization; CV kernel mapping (Mix kernel CV fusion, inter-core sync, CVPipeline, AutoSubTiling), on-chip memory mapping, and multi-stage pipeline/instruction mapping.  
  - **HACC**: Heterogeneous hardware abstraction for Host/Device programming and launch semantics; **Annotation**, **Scope** for compiler hints and scope marking.

- **Key compilation features**  
  CV fusion and pipelining (CVPipeline, AutoSubTiling), automatic memory planning (PlanMemory) and pipeline sync (AutoSync), blockization and scheduling (AutoBlockify, AutoFlatten, AutoSchedule), custom ops, DFX, and CV optimizations for portable performance while preserving high-level semantics.

- **Ecosystem and openness**  
  Layered interfaces for PyTorch (Torch-MLIR), TileLang, Triton, and other frameworks, balancing performance and usability to enable Ascend AI processors.

## Next steps

- [Install and build](quick_start/installing_guide.md) — Environment and build
- [Quick start](quick_start/index.rst) — Examples and entry
- [Architecture](architecture.md) — Logical and code architecture, compilation flow
