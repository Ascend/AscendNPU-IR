# 相关项目

## [MLIR](https://mlir.llvm.org)
MLIR 源自 LLVM 社区，提供了一套革命性的、可复用和可扩展的编译器基础设施。AscendNPU IR构建在MLIR框架之上，充分利用了其核心优势：
- **模块化设计**：可以定义不同抽象层次的IR，方便进行渐进式 lowering。
- **基础设施复用**：直接使用 MLIR 提供的解析、转换、优化和代码生成工具链，避免了重复造轮子。
- **生态互通性**：扩展 MLIR 方言，理论上能与 MLIR 生态中的其他方言（如 TensorFlow、PyTorch 导出的 IR）进行交互和转换，为连接上层框架提供了潜在通路。

## [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)
Triton-Ascend是面向昇腾平台构建的Triton编译框架，旨在让Triton代码能够在昇腾硬件上高效运行。AscendNPU IR 通过支持作为 Triton 的一个编译后端，使得开发者可以使用熟悉的 Triton 语法和编程模型，直接为昇腾NPU编写高性能内核代码，极大地降低了Python开发者为昇腾编写高性能算子的门槛。

## [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)
TileLang 是一个用于描述张量计算的领域特定语言。TileLang-Ascend 是其针对昇腾的版本。通过将 AscendNPU IR 作为其编译目标，TileLang-Ascend 能够利用 AscendNPU IR 的硬件特化优化能力，生成高效的昇腾内核。

