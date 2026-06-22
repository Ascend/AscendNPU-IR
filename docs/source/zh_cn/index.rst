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

- :doc:`安装与构建 <introduction/quick_start/installing_guide>` — 环境要求与编译步骤
- :doc:`快速开始 <introduction/quick_start/index>` — 安装说明与示例
- :doc:`简介与架构 <introduction/architecture>` — 逻辑架构与代码结构

用户指南
---------------------

- :doc:`编译选项 <user_guide/compile_option>` - 编译选项与功能说明
- :doc:`调试调测 <user_guide/debug_option>` - 调试方法介绍
- :doc:`最佳实践 <user_guide/best_practice>` - 编程案例详解与算子改写

开发者指南
----------------------------

- :doc:`IR 接入指南 <developer_guide/conversion/interface_api>` — 生态对接与接口
- :doc:`Dialects <developer_guide/dialects/index>` — 方言说明
- :doc:`Passes <developer_guide/passes/index>` — 变换与 Pass
- :doc:`关键特性 <developer_guide/features/index>` — 特性与实现说明

更多
------------

- :doc:`贡献指南 <contributing_guide/contribute>`
- :doc:`常见问题 <faq/faq>`
- :doc:`相关项目与资源 <reference/thanks>`
- :doc:`讲座与课程 <reference/talk_and_course>`


.. toctree 驱动侧栏导航。

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 入门指南

   项目简介 <introduction/introduction>
   快速开始 <introduction/quick_start/index>
   架构设计 <introduction/architecture>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 用户指南

   编译选项 <user_guide/compile_option>
   调试调测 <user_guide/debug_option>
   最佳实践 <user_guide/best_practice>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 开发者指南

   接入指南 <developer_guide/conversion/index>
   Dialects <developer_guide/dialects/index>
   Passes <developer_guide/passes/index>
   关键特性 <developer_guide/features/index>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 贡献与支持

   贡献指南 <contributing_guide/contribute>
   AscendNPU IR 用户 <user_of_npuir/users>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 常见问题

   常见问题(FAQ) <faq/faq>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: 参考资源

   相关项目与致谢 <reference/thanks>
   讲座与课程 <reference/talk_and_course>

