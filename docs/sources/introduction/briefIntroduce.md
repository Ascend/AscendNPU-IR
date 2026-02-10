# 简介

AscendNPU IR（AscendNPU Intermediate Representation）是基于MLIR（Multi-Level Intermediate Representation）构建的，面向昇腾亲和算子编译时使用的中间表示，提供昇腾完备表达能力，通过编译优化提升昇腾AI处理器计算效率，支持通过生态框架使能昇腾AI处理器与深度调优。

AscendNPU IR提供多级抽象接口：提供一系列高层抽象接口，屏蔽昇腾计算、搬运、同步指令细节，编译优化自动感知硬件架构，将硬件无关表达映射到底层指令，提升算子开发易用性；同时提供细粒度性能控制接口，能够精准控制片上内存地址、流水同步插入位置以及是否使能乒乓流水优化等，允许性能细粒度控制。

AscendNPU IR通过开源社区开放接口，支持生态框架灵活对接，高效使能昇腾AI处理器。