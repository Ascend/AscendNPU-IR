# 简介

**AscendNPU IR**（AscendNPU Intermediate Representation）是基于 MLIR（Multi-Level Intermediate Representation）构建的面向昇腾亲和的算子编译中间表示，提供昇腾完备表达能力，通过编译优化提升昇腾AI处理器计算效率，通过开源社区开放接口，支持生态框架灵活对接，高效使能昇腾AI处理器。

---

## 关键能力

- **多级抽象与易用性**  
提供高层抽象接口，屏蔽昇腾计算、搬运、同步等指令细节，编译器自动感知硬件架构并将硬件无关表达映射到底层指令；同时提供细粒度性能控制接口，可精准控制片上内存布局、流水同步插入位置、是否使能乒乓流水等，兼顾易用与性能调优。
- **分层方言与编译优化**  
  - **HFusion**：基于 Linalg 扩展，负责硬件相对无关的优化与生态对接；支持与 Arith、Math、Torch 等方言的转换，以及 Tensor 化简、类型合法化、算子融合生成。  
  - **HIVM**：面向昇腾对计算、搬运、同步进行 Tile 级抽象，屏蔽底层指令参数；负责 CV 核映射（Mix Kernel 的 CV 融合、核间同步、CVPipeline 流水与 AutoSubTiling 等）、核内片上内存映射与多级流水/指令映射。  
  - **HACC**：异构硬件抽象，表达 Host/Device 编程模型与 launch 语义；**Annotation**、**Scope** 等用于 compiler hint 与作用域标记。
- **关键编译特性**  
支持 CV 融合与流水（CVPipeline、AutoSubTiling）、自动内存规划（PlanMemory）与流水同步（AutoSync）、块化与调度（AutoBlockify、AutoFlatten、AutoSchedule），以及自定义算子、DFX、CV 优化等，便于在保持高层语义的前提下获得可移植性能。
- **生态对接与开放**  
通过分层接口支持与 PyTorch（Torch-MLIR）、TileLang、Triton 及各类框架的对接，高性能与易用性灵活平衡，高效使能昇腾 AI 处理器。

---

## 下一步

- :doc:`安装与构建 <sources/introduction/quick_start/installing_guide_zh>` — 环境与编译  
- :doc:`快速开始 <sources/introduction/quick_start/index_zh>` — 示例与使用入口  
- :doc:`架构设计 <sources/introduction/architecture_zh>` — 逻辑架构、代码架构与编译流程  
- :doc:`编程模型 <sources/introduction/programming_model_zh>` — Host/Device 与 Tile 编程模型
