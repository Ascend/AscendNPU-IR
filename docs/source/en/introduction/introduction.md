# Introduction

**AscendNPU IR** (AscendNPU Intermediate Representation) is an Ascend-affinity operator compilation intermediate representation built on **MLIR** (Multi-Level Intermediate Representation). It provides complete expressiveness for Ascend, improves the computational efficiency of Ascend AI Processors through compilation optimization, and, by opening interfaces through the open-source community, supports flexible integration with ecosystem frameworks to efficiently enable Ascend AI Processors.

## Key Capabilities

- **Multi-level Abstraction and Usability**

  Provides high-level abstraction interfaces that hide the details of Ascend computation, data movement, and synchronization instructions. The compiler automatically senses the hardware architecture and maps hardware-independent expressions to low-level instructions. It also provides fine-grained performance control interfaces for precisely controlling on-chip memory layout, pipeline synchronization insertion positions, and whether to enable ping-pong pipelining, balancing ease of use with performance tuning.

- **Layered Dialects and Compilation Optimization**
    - **HFusion**: Built on Linalg extensions, it handles hardware-independent optimizations and ecosystem integration. It supports conversions with dialects such as Arith, Math, and Torch, as well as tensor simplification, type legalization, and operator fusion generation.
    - **HIVM**: Provides Tile-level abstraction of computation, data movement, and synchronization for Ascend, hiding low-level instruction parameters. It handles CV kernel mapping (CV fusion of Mix Kernel, inter-core synchronization, CVPipeline pipelining, AutoSubTiling, and more), on-chip memory mapping within kernels, and multi-level pipelining/instruction mapping.
    - **HACC**: Heterogeneous hardware abstraction that expresses the Host/Device programming model and launch semantics. Annotation, Scope, and others are used for `compiler hint` and scope marking.

- **Key Compilation Features**

  Supports CV fusion and pipelining (CVPipeline, AutoSubTiling), automatic memory planning (PlanMemory) and pipeline synchronization (AutoSync), blocking and scheduling (AutoBlockify, AutoFlatten, AutoSchedule), as well as custom operators, DFX, and CV optimization, enabling portable performance while preserving high-level semantics.

- **Ecosystem Integration and Openness**

Through layered interfaces, it supports integration with PyTorch (Torch-MLIR), TileLang, Triton, and various frameworks, flexibly balancing high performance and ease of use to efficiently enable the Ascend AI Processor.

## Next Steps

- [Installation and Build](quick_start/installing_guide.md) — Environment and Compilation
- [Quick Start](quick_start/index.rst) — Examples and Usage Entry
- [Architecture Design](architecture.md) — Logical Architecture, Code Architecture, and Compilation Process
