# 相关项目与致谢

本文列出与AscendNPU IR密切相关的开源项目与生态，并致谢LLVM/MLIR等社区。

## [MLIR](https://mlir.llvm.org)

MLIR源自LLVM社区，提供可复用、可扩展的编译器基础设施。AscendNPU IR构建在MLIR之上，感谢LLVM/MLIR社区所有开发者与贡献者。AscendNPU IR充分利用MLIR的以下优势：

- **模块化设计**：可定义不同抽象层次的IR，便于渐进式lowering。
- **基础设施复用**：复用MLIR的解析、转换、优化与代码生成工具链。
- **生态互通性**：通过扩展MLIR方言，可与MLIR生态中其他方言（如PyTorch、TensorFlow导出的IR）交互与转换，为对接上层框架提供通路。

## [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)

Triton-Ascend是面向昇腾平台构建的Triton编译框架，使Triton代码能在昇腾硬件上高效运行。AscendNPU IR作为Triton的编译后端，使开发者能以熟悉的Triton语法与编程模型为昇腾NPU高效开发kernel，降低Python开发者为昇腾开发算子的门槛。

## [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)

TileLang是用于描述张量计算的领域特定语言，TileLang-Ascend为其面向昇腾的版本。通过将AscendNPU IR作为编译后端，TileLang-Ascend可借助AscendNPU IR的昇腾硬件亲和优化能力生成高性能昇腾算子。
