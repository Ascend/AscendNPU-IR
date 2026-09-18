<div align="center">

# AscendNPU-IR

### MLIR for Ascend, delivering comprehensive support

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://ascendnpu-ir.gitcode.com)
[![license](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/LICENSE)
[![contributing](https://img.shields.io/badge/CONTRIBUTING-teal)](https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/contributing_guide/contribute.md)
[![SIG](https://img.shields.io/badge/SIG-AscendNPU-IR)](https://etherpad.ascend.osinfra.cn/p/sig-AscendNPU-IR)
</div>

## 🎯 Introduction

![AscendNPU IR README](./docs/source/images/introduction/ascendnpu-ir-in-cann.png "ascendnpu-ir-in-cann.png")

AscendNPU IR (AscendNPU Intermediate Representation) is built on MLIR (Multi-Level Intermediate Representation). It serves as an intermediate representation for compiling Ascend-compatible operators and provides comprehensive Ascend expression capabilities. It enhances the computational efficiency of Ascend AI processors through compilation optimization and supports deep tuning via ecosystem frameworks.

AscendNPU IR offers multi-level abstraction interfaces: it provides a series of high-level abstraction interfaces encapsulate Ascend computation, data movement, and synchronization instructions. The compilation optimizations automatically detect hardware architecture, map hardware-agnostic expressions to underlying instructions to enhance operator development ease. Additionally, it provides fine-grained performance control interfaces, enabling precise control over on-chip memory addresses, pipeline synchronization insertion points, ping-pong pipeline optimization activation, and allowing granular performance control.

AscendNPU IR supports flexible integration with ecosystem frameworks and efficiently enabling Ascend AI processors through open interfaces via the open-source community.

## 🔗 Repository & Documentation

- **GitCode:** [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- **GitHub:** [AscendNPU-IR](https://github.com/Ascend/AscendNPU-IR)
- **Documentation:** [Documentation](https://ascendnpu-ir.gitcode.com)

## 🔍 Repository Structure

Key directories within the AscendNPU IR repository are as follows:

```plaintext
├── bishengir            // Source code directory
│   ├── cmake
│   ├── include          // Header files
│   ├── lib              // Source file
│   ├── test             // Test cases
│   |  └── Integration   // E2E use cases
│   └── tools            // Binary tools
├── build-tools          // Build tools
├── CMakeLists.txt
├── docs                 // Documentation
├── LICENSE
├── NOTICE
├── README.md
└── README_zh.md
```

## 📚 Documentation

AscendNPU IR uses a Sphinx-based documentation project under `docs/`, with **dual-language support (English & Chinese)** through paired Markdown sources.

- **Language**: see `docs/README.md` (English) and `docs/README_zh.md` (Chinese) for full details.
- **File scheme**:
  - **English**: The entry page is `docs/source/en/index.rst`; content under `docs/source/en` directory.
  - **Chinese**: The entry page is `docs/source/zh_cn/index.rst`; content under `docs/source/zh_cn` directory.
- **Naming convention**: Directory names and document file names under `docs/` (including `docs/source/`) adopt a unified lowercase **snake_case** style, for example: `quick_start/`, `installing_guide.md`, `user_guide/`，to keep consistent paths and URLs.

To build the documentation from the repository root:

```bash
make -C docs html      # English only → docs/_build/en
make -C docs html-zh   # Chinese only → docs/_build/zh_cn
make -C docs html-all  # Both languages
```

For more options (local preview, Read the Docs deployment, adding new documents, and doc layout), please refer to `docs/README.md` (or `docs/README_zh.md` for Chinese).

## ⚡️ Quick Start

Quick Start Guide: [Quick Start](./docs/source/en/introduction/quick_start/index.rst)

For build and installation instructions, see: [Build and Installation](./docs/source/en/introduction/quick_start/installing_guide.md)

For an example of building E2E cases, see: [README.md](./bishengir/test/Integration/README.md)

| Example Name | Guide |
|------|------|
| HIVM VecAdd |  [VecAdd README.md](./bishengir/test/Integration/HIVM/VecAdd/README.md) |

## 📝 Version Compatibility Notes

Please refer to the relevant sections of the [CANN Community Edition Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0001.html) for installation and preparation of Ascend hardware, CANN software, and corresponding deep learning frameworks.

## 📄 License

[Apache License v2.0](LICENSE)
