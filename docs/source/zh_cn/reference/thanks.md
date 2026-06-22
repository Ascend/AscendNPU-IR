# 相关项目与致谢

本文列出与 AscendNPU IR 密切相关的开源项目与生态，并致谢 LLVM/MLIR 等社区。

## [MLIR](https://mlir.llvm.org)

MLIR 源自 LLVM 社区，提供可复用、可扩展的编译器基础设施。AscendNPU IR 构建在 MLIR 之上，感谢 LLVM/MLIR 社区所有开发者与贡献者。AscendNPU IR 充分利用 MLIR 的以下优势：

- **模块化设计**：可定义不同抽象层次的 IR，便于渐进式 lowering。
- **基础设施复用**：复用 MLIR 的解析、转换、优化与代码生成工具链。
- **生态互通性**：通过扩展 MLIR 方言，可与 MLIR 生态中其他方言（如 TensorFlow、PyTorch 导出的 IR）交互与转换，为对接上层框架提供通路。

## [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)

Triton-Ascend 是面向昇腾平台构建的Triton编译框架，使 Triton 代码能在昇腾硬件上高效运行。AscendNPU IR 作为 Triton 的编译后端，使开发者能以熟悉的 Triton 语法与编程模型为昇腾 NPU 高效开发kernel，降低 Python 开发者为昇腾开发算子的门槛。

## [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)

TileLang 是用于描述张量计算的领域特定语言，TileLang-Ascend 为其面向昇腾的版本。通过将 AscendNPU IR 作为编译后端，TileLang-Ascend 可借助 AscendNPU IR 的昇腾硬件亲和优化能力生成高性能昇腾算子。
