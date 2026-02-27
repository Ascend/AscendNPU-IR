.. AscendNPU IR documentation master file.
   Keep the root toctree for sidebar navigation; content below is the homepage.

Welcome to AscendNPU IR 文档
============================

**AscendNPU IR** 是基于 MLIR 的昇腾亲和算子编译中间表示（Intermediate Representation），面向昇腾 AI 处理器提供多级抽象与编译优化能力，支持生态框架灵活对接，在保证易用性的同时支持细粒度性能调优。

`GitCode 仓库 <https://gitcode.com/Ascend/AscendNPU-IR>`_ · *当前文档为非正式版*

Getting Started / 入门
----------------------

- :doc:`安装与构建 <sources/introduction/quickStart/installingGuide>` — 环境要求与编译步骤
- :doc:`快速开始 <sources/introduction/quickStart/index>` — 安装说明与示例
- :doc:`简介与架构 <sources/introduction/architecture>` — 逻辑架构与代码结构
- :doc:`编程模型 <sources/introduction/programmingModel>` — 编程模型说明

User Guide / 用户指南
---------------------

- :doc:`编译选项 <sources/user-guide/compileOption>`
- :doc:`调试调测 <sources/user-guide/debugOption>`
- :doc:`最佳实践 <sources/user-guide/bestPractice>`

Developer Guide / 开发者指南
----------------------------

- :doc:`IR 接入指南 <sources/developer-guide/conversion/interfaceAPI>` — 生态对接与接口
- :doc:`Dialects <sources/developer-guide/dialects/index>` — 方言说明
- :doc:`Passes <sources/developer-guide/passes/index>` — 变换与 Pass
- :doc:`关键特性 <sources/developer-guide/features/index>` — 特性与实现说明

About / 更多
------------

- :doc:`贡献指南 <sources/contributing-guide/contribute>`
- :doc:`常见问题 <sources/FAQ/FAQ>`
- :doc:`相关项目与资源 <sources/reference/thanks>`
- :doc:`讲座与课程 <sources/reference/talkAndCourse>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Introduction

   sources/introduction/briefIntroduce.md
   sources/introduction/quickStart/index
   sources/introduction/programmingModel.md
   sources/introduction/architecture.md

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   sources/user-guide/compileOption.md
   sources/user-guide/debugOption.md
   sources/user-guide/bestPractice.md

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   sources/developer-guide/conversion/index
   sources/developer-guide/dialects/index
   sources/developer-guide/passes/index
   sources/developer-guide/features/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing

   sources/contributing-guide/contribute.md
   sources/userOfnpuir/users.md

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FAQ

   sources/FAQ/FAQ.md

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Reference

   sources/reference/thanks.md
   sources/reference/talkAndCourse.md
