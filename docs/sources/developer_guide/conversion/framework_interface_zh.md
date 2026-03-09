# 框架接入

AscendNPU IR 是基于 MLIR 的编译框架。其向上支持与 **Triton**、**TileLang** 等语言或框架的对接，从而使能三方语言或框架支持昇腾硬件，在 NPU 上运行自定义算子。

本文档仅作简介说明，并给出本目录下各接入文档的索引。

## 本目录接入说明索引

| 接入方式 | 说明 | 文档 |
|----------|------|------|
| **Triton** | 使用 Triton 编写高性能内核，通过 Triton Ascend 在昇腾 NPU 上运行。含安装、环境、算子映射及昇腾扩展说明。 | [Triton 接入](triton_interface_zh.md) |
| **TileLang** | 使用 TileLang Ascend（基于 tile-lang/TVM 的 DSL）开发面向昇腾 NPU 的内核（如 GEMM、向量运算、attention）。含环境、构建与快速开始。 | [TileLang 接入](tile_lang_interface_zh.md) |

关于 IR 层概念、公共编译选项及其他框架接入路径（如 Torch、Linalg、HIVM），请参阅 [IR 接入简介](interface_api_zh.md)。
