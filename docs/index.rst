.. AscendNPU IR documentation master file (English).
   Keep the root toctree for sidebar navigation.

Welcome to AscendNPU IR Docs
============================

**AscendNPU IR** is an MLIR-based intermediate representation for Ascend NPU operator compilation, providing multi-level abstraction and compiler optimizations, with flexible integration for ecosystem frameworks and fine-grained performance tuning.

.. raw:: html

    <p><a href="https://gitcode.com/Ascend/AscendNPU-IR" target="_blank">AscendNPU-IR on GitCode</a></p>
    <p><a href="https://github.com/Ascend/AscendNPU-IR" target="_blank">AscendNPU-IR on GitHub</a></p>
    <p><a href="https://ascendnpu-ir.readthedocs.io/" target="_blank">Documentation on Read the Docs</a></p>

Getting Started
---------------

- :doc:`Install and build <sources/introduction/quick_start/installing_guide>` — Environment and build steps
- :doc:`Quick start <sources/introduction/quick_start/index>` — Setup and examples
- :doc:`Introduction and architecture <sources/introduction/architecture>` — Architecture and layout
- :doc:`Programming model <sources/introduction/programming_model>` — Programming model

User Guide
----------

- :doc:`Compile options <sources/user_guide/compile_option>`
- :doc:`Debug and tune <sources/user_guide/debug_option>`
- :doc:`Best practices <sources/user_guide/best_practice>`

Developer Guide
---------------

- :doc:`IR integration <sources/developer_guide/conversion/interface_api>` — Ecosystem and APIs
- :doc:`Dialects <sources/developer_guide/dialects/index>` — Dialect reference
- :doc:`Passes <sources/developer_guide/passes/index>` — Passes and transforms
- :doc:`Features <sources/developer_guide/features/index>` — Key features

About
-----

- :doc:`Contributing <sources/contributing_guide/contribute>`
- :doc:`FAQ <sources/faq/faq>`
- :doc:`Related projects <sources/reference/thanks>`
- :doc:`Talks and courses <sources/reference/talk_and_course>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Introduction

   sources/introduction/introduction
   sources/introduction/quick_start/index
   sources/introduction/programming_model
   sources/introduction/architecture

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   sources/user_guide/compile_option
   sources/user_guide/debug_option
   sources/user_guide/best_practice

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   sources/developer_guide/conversion/index
   sources/developer_guide/dialects/index
   sources/developer_guide/passes/index
   sources/developer_guide/features/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contributing

   sources/contributing_guide/contribute
   sources/user_of_npuir/users

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: FAQ

   sources/faq/faq

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Reference

   sources/reference/thanks
   sources/reference/talk_and_course
