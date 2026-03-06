.. AscendNPU IR 文档主入口（中文）
   toctree 驱动侧栏导航。

Welcome to AscendNPU IR 文档
============================

**AscendNPU IR** 是基于 MLIR 的昇腾亲和算子编译中间表示（Intermediate Representation），面向昇腾 AI 处理器提供多级抽象与编译优化能力，支持生态框架灵活对接，在保证易用性的同时支持细粒度性能调优。

- `GitCode 仓库 <https://gitcode.com/Ascend/AscendNPU-IR>`_
- `GitHub 仓库 <https://github.com/Ascend/AscendNPU-IR>`_
- `AscendNPU IR 文档 <https://npuir.readthedocs.io/>`_

Getting Started
----------------------

- :doc:`安装与构建 <sources/introduction/quick_start/installing_guide_zh>` — 环境要求与编译步骤
- :doc:`快速开始 <sources/introduction/quick_start/index_zh>` — 安装说明与示例
- :doc:`简介与架构 <sources/introduction/architecture_zh>` — 逻辑架构与代码结构
- :doc:`编程模型 <sources/introduction/programming_model_zh>` — 编程模型说明

User Guide / 用户指南
---------------------

- :doc:`编译选项 <sources/user_guide/compile_option_zh>` — 编译选项与功能说明
- :doc:`调试调测 <sources/user_guide/debug_option_zh>` — 调试方法介绍
- :doc:`最佳实践 <sources/user_guide/best_practice_zh>` — 编程案例详解与算子改写

Developer Guide / 开发者指南
----------------------------

- :doc:`IR 接入指南 <sources/developer_guide/conversion/interface_api_zh>` — 生态对接与接口
- :doc:`Dialects <sources/developer_guide/dialects/index_zh>` — 方言说明
- :doc:`Passes <sources/developer_guide/passes/index_zh>` — 变换与 Pass
- :doc:`关键特性 <sources/developer_guide/features/index_zh>` — 特性与实现说明

About / 更多
------------

- :doc:`贡献指南 <sources/contributing_guide/contribute_zh>`
- :doc:`常见问题 <sources/faq/faq_zh>`
- :doc:`相关项目与资源 <sources/reference/thanks_zh>`
- :doc:`讲座与课程 <sources/reference/talk_and_course_zh>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Introduction

   sources/introduction/introduction_zh
   sources/introduction/quick_start/index_zh
   sources/introduction/programming_model_zh
   sources/introduction/architecture_zh

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   sources/user_guide/compile_option_zh
   sources/user_guide/debug_option_zh
   sources/user_guide/best_practice_zh

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   sources/developer_guide/conversion/index_zh
   sources/developer_guide/dialects/index_zh
   sources/developer_guide/passes/index_zh
   sources/developer_guide/features/index_zh

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing

   sources/contributing_guide/contribute_zh
   sources/user_of_npuir/users_zh

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FAQ

   sources/faq/faq_zh

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Reference

   sources/reference/thanks_zh
   sources/reference/talk_and_course_zh
