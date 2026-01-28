.. Triton-Ascend documentation master file, created by
   sphinx-quickstart on Wed Nov 26 09:25:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AscendNPU IR documentation!
=========================================

`AscendNPU-IR <https://gitcode.com/Ascend/AscendNPU-IR/tree/master>`_ 是面向昇腾亲和算子编译时使用的中间表示。

目前本文档为非正式版。


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
   sources/developer-guide/dialect/index
   sources/developer-guide/passes.md
   sources/developer-guide/features.md

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
   :maxdepth: 2
   :hidden:
   :caption: Reference

   sources/reference/thanks.md
   sources/reference/talkAndCourse.md


Getting Started
---------------
- Follow the :doc:`installation instructions <sources/quick-start/installing-guide>` to build AscendNPU IR.

Developer Guide
---------------
- :doc:`IR接入指南 <sources/developer-guide/interfaceAPI>`
- :doc:`Dialects <sources/developer-guide/dialect>`
- :doc:`调试指南 <sources/developer-guide/debug_guide>`
- :doc:`性能优化实践 <sources/developer-guide/performance>`