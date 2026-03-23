.. AscendNPU IR documentation master file (English).

Welcome to AscendNPU IR Docs
============================

**AscendNPU IR** is an MLIR-based intermediate representation for Ascend NPU operator compilation, providing multi-level abstraction and compiler optimizations, with flexible integration for ecosystem frameworks and fine-grained performance tuning.

.. raw:: html

    <ul>
    <li><a href="https://gitcode.com/Ascend/AscendNPU-IR" target="_blank">AscendNPU-IR on GitCode</a></li>
    <li><a href="https://github.com/Ascend/AscendNPU-IR" target="_blank">AscendNPU-IR on GitHub</a></li>
    <li><a href="https://ascendnpu-ir.gitcode.com" target="_blank">Documentation on GitCode</a></li>
    </ul>

Getting Started
---------------

- :doc:`Install and Build <sources/introduction/quick_start/installing_guide>` — Environment requirements and build steps
- :doc:`Quick start <sources/introduction/quick_start/index>` — Setup instructions and examples
- :doc:`Architecture Design <sources/introduction/architecture>` — Logical architecture and code structure

User Guide
----------

- :doc:`Compile Options <sources/user_guide/compile_option>` — Compilation options and features
- :doc:`Debug and Tune <sources/user_guide/debug_option>` — Debugging methods and tuning
- :doc:`Best Practices <sources/user_guide/best_practice>` — Programming cases and operator rewriting

Developer Guide
---------------

- :doc:`IR integration <sources/developer_guide/conversion/interface_api>` — Ecosystem integration and APIs
- :doc:`Dialects <sources/developer_guide/dialects/index>` — Dialect reference
- :doc:`Passes <sources/developer_guide/passes/index>` — Transforms and passes
- :doc:`Key Features <sources/developer_guide/features/index>` — Features and implementation details

About
-----

- :doc:`Contributing <sources/contributing_guide/contribute>`
- :doc:`FAQ <sources/faq/faq>`
- :doc:`Related Projects and Acknowledgments <sources/reference/thanks>`
- :doc:`Talks and Courses <sources/reference/talk_and_course>`


.. Keep the root toctree for sidebar navigation.

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Introduction

   Project Overview <sources/introduction/introduction>
   Quick Start <sources/introduction/quick_start/index>
   Architecture Design <sources/introduction/architecture>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: User Guide

   Compile Options <sources/user_guide/compile_option>
   Debug and Tune <sources/user_guide/debug_option>
   Best Practices <sources/user_guide/best_practice>   

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Developer Guide

   Conversion Guide <sources/developer_guide/conversion/index>
   Dialects <sources/developer_guide/dialects/index>
   Passes <sources/developer_guide/passes/index>
   Key Features <sources/developer_guide/features/index>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Contributing

   Contributing Guide <sources/contributing_guide/contribute>
   AscendNPU IR Users <sources/user_of_npuir/users>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: FAQ

   FAQ <sources/FAQ/FAQ>

.. toctree::
   :hidden:
   :titlesonly:
   :caption: Reference

   Related Projects and Acknowledgments <sources/reference/thanks>
   Talks and Courses <sources/reference/talk_and_course>