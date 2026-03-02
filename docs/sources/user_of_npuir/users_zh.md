# AscendNPU IR 用户

本项目已使能多个算子编程框架接入昇腾后端，并提供面向昇腾的编译优化能力。以下为已对接或使用 AscendNPU IR 的语言与框架示例。

## 语言生态

| DSL | 简介 |
| --- | --- |
| [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend) | 为 Triton 开发者提供面向昇腾的算子快速开发与生态迁移能力 |
| [TileLang-Ascend (branch npuir)](https://github.com/tile-ai/tilelang-ascend/tree/npuir) | 简化高性能算子开发流程的 Tile 级编程框架，兼顾开发效率与底层优化能力 |
| [DLCompiler](https://github.com/DeepLink-org/DLCompiler) | 扩展 Triton 的深度学习编译器，支持跨架构 DSL 扩展与智能自动优化 |
| [FlagTree](https://github.com/flagos-ai/flagtree) | 基于 Triton 语言的开源 AI 编译器，提供跨多后端的统一编译能力 |
