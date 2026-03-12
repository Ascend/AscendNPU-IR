![AscendNPU IR定位](./docs/pic/ascendnpu-ir-in-cann.png "ascendnpu-ir-in-cann.png")

## 🎯 项目介绍

AscendNPU IR（AscendNPU Intermediate Representation）是基于MLIR（Multi-Level Intermediate Representation）构建的，面向昇腾亲和算子编译时使用的中间表示，提供昇腾完备表达能力，通过编译优化提升昇腾AI处理器计算效率，支持通过生态框架使能昇腾AI处理器与深度调优。

AscendNPU IR提供多级抽象接口：提供一系列高层抽象接口，屏蔽昇腾计算、搬运、同步指令细节，编译优化自动感知硬件架构，将硬件无关表达映射到底层指令，提升算子开发易用性；同时提供细粒度性能控制接口，能够精准控制片上内存地址、流水同步插入位置以及是否使能乒乓流水优化等，允许性能细粒度控制。

AscendNPU IR通过开源社区开放接口，支持生态框架灵活对接，高效使能昇腾AI处理器。

## 🔗 仓库与文档

- **GitCode:** [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- **GitHub:** [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR)
- **文档:** [文档](https://ascendnpu-ir.gitcode.com)

## 🔍 仓库结构
AscendNPU IR仓关键目录如下所示：
```
├── bishengir            // 源码目录
│   ├── cmake
│   ├── include          // 头文件
│   ├── lib              // 源文件
│   ├── test             // 测试用例
│   |  └── Integration   // 端到端用例
│   └── tools            // 二进制工具
├── build-tools          // 构建工具
├── CMakeLists.txt
├── docs                 // 文档
├── LICENSE
├── NOTICE
├── README.md
└── README_zh.md
```

## 📚 文档说明

本项目在 `docs/` 目录下使用基于 Sphinx 的文档工程，通过成对的 Markdown 源文件提供**英文与中文双语文档**。

- **Language / 语言**：完整文档方案请参见 `docs/README.md`（English）与 `docs/README_zh.md`（中文）。
- **文档组织方案**：
  - **英文**：默认 `.md` 文件，入口页为 `docs/index.rst`，正文位于 `docs/sources/**/*.md`。
  - **中文**：`*_zh.md` 文件，入口页为 `docs/index_zh.rst`，正文位于 `docs/sources/**/*_zh.md`。
- **命名规范**：在 `docs/`（包含 `docs/sources/`）下，**目录名**与**文档文件名**统一采用 **snake_case**，例如：`quick_start/`、`installing_guide.md`、`installing_guide_zh.md`、`user_guide/`、`developer_guide/`、`contributing_guide/`，以保持路径与 URL 风格一致。

在仓库根目录下构建文档：

```bash
make -C docs html      # 仅英文 → docs/_build/en
make -C docs html-zh   # 仅中文 → docs/_build/zh_cn
make -C docs html-all  # 中英文均构建
```

关于本地预览、Read the Docs 部署、新增文档与整体结构说明，请参考 `docs/README_zh.md`（或英文版 `docs/README.md`）。

## ⚡️ 快速上手

快速上手指南请见：[快速上手](./docs/sources/introduction/quick_start/index_zh.md)



编译与安装指南请见：[构建安装](./docs/sources/introduction/quick_start/installing_guide_zh.md)

构建端到端用例示例请见：[README_zh.md](./bishengir/test/Integration/README_zh.md)

| 示例名称 | 构建指南 |
|------|------|
| HIVM VecAdd |  [VecAdd README_zh.md](./bishengir/test/Integration/HIVM/VecAdd/README_zh.md) |

## 📝 版本配套说明
请参考[CANN社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0001.html)相关章节，对昇腾硬件、CANN软件及相应深度学习框架进行安装准备。

## 📄 许可证书
[Apache License v2.0](LICENSE)
