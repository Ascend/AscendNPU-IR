.. AscendNPU IR 中文文档主入口

欢迎查看 AscendNPU IR 文档
============================

**AscendNPU IR** 是基于 MLIR 的昇腾亲和算子编译中间表示（Intermediate Representation），面向昇腾 AI 处理器提供多级抽象与编译优化能力，支持生态框架灵活对接，在保证易用性的同时支持细粒度性能调优。

.. raw:: html

    <ul>
    <li><a href="https://gitcode.com/Ascend/AscendNPU-IR" target="_blank">GitCode 仓库</a></li>
    <li><a href="https://github.com/Ascend/AscendNPU-IR" target="_blank">GitHub 仓库</a></li>
    <li><a href="https://ascendnpu-ir.gitcode.com" target="_blank">AscendNPU IR 文档</a></li>
    </ul>

快速入门
----------------------

- :doc:`安装与构建 <sources/introduction/quick_start/installing_guide_zh>` — 环境要求与编译步骤
- :doc:`快速开始 <sources/introduction/quick_start/index_zh>` — 安装说明与示例
- :doc:`简介与架构 <sources/introduction/architecture_zh>` — 逻辑架构与代码结构

用户指南
---------------------

- :doc:`编译选项 <sources/user_guide/compile_option_zh>` - 编译选项与功能说明
- :doc:`调试调测 <sources/user_guide/debug_option_zh>` - 调试方法介绍
- :doc:`最佳实践 <sources/user_guide/best_practice_zh>` - 编程案例详解与算子改写

开发者指南
----------------------------

- :doc:`IR 接入指南 <sources/developer_guide/conversion/interface_api_zh>` — 生态对接与接口
- :doc:`Dialects <sources/developer_guide/dialects/index_zh>` — 方言说明
- :doc:`Passes <sources/developer_guide/passes/index_zh>` — 变换与 Pass
- :doc:`关键特性 <sources/developer_guide/features/index_zh>` — 特性与实现说明

更多
------------

- :doc:`贡献指南 <sources/contributing_guide/contribute_zh>`
- :doc:`常见问题 <sources/faq/faq_zh>`
- :doc:`相关项目与资源 <sources/reference/thanks_zh>`
- :doc:`讲座与课程 <sources/reference/talk_and_course_zh>`


.. toctree 驱动侧栏导航。

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 入门指南

   项目简介 <sources/introduction/introduction_zh>
   快速开始 <sources/introduction/quick_start/index_zh>
   架构设计 <sources/introduction/architecture_zh>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 用户指南

   编译选项 <sources/user_guide/compile_option_zh>
   调试调测 <sources/user_guide/debug_option_zh>
   最佳实践 <sources/user_guide/best_practice_zh>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 开发者指南

   接入指南 <sources/developer_guide/conversion/index_zh>
   Dialects <sources/developer_guide/dialects/index_zh>
   Passes <sources/developer_guide/passes/index_zh>
   关键特性 <sources/developer_guide/features/index_zh>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 贡献与支持

   贡献指南 <sources/contributing_guide/contribute_zh>
   AscendNPU IR 用户 <sources/user_of_npuir/users_zh>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 常见问题

   常见问题(FAQ) <sources/faq/faq_zh>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 参考资源

   相关项目与致谢 <sources/reference/thanks_zh>
   讲座与课程 <sources/reference/talk_and_course_zh>

