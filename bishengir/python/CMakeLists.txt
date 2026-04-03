# Disables generation of "version soname" (i.e. libFoo.so.<version>), which
# causes pure duplication as part of Python wheels.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(BISHENGIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bishengir")

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=bishengir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(BiShengIRPythonSources)
declare_mlir_python_sources(BiShengIRPythonExtensions)

declare_mlir_python_sources(BiShengIRPythonSources.Dialects
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT BiShengIRPythonSources
)

declare_mlir_python_sources(BiShengIRPythonSources.Core
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT BiShengIRPythonSources
)

declare_mlir_python_sources(BiShengIRPythonSources.Core.Python
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT BiShengIRPythonSources.Core
  SOURCES
    bishengtile.py
    helper.py
)

declare_mlir_python_sources(BiShengIRSiteInitialize
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT BiShengIRPythonSources
  SOURCES
    _mlir_libs/_site_initialize_0.py
)

################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BiShengIRPythonSources.Dialects
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/HIVMOps.td
  SOURCES_GLOB dialects/hivm.py
  DIALECT_NAME hivm
  GEN_ENUM_BINDINGS_TD_FILE
  "dialects/HIVMAttrs.td"
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BiShengIRPythonSources.Dialects
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/HFusionOps.td
  SOURCES_GLOB dialects/hfusion.py
  DIALECT_NAME hfusion
  DEPENDS HFusionOdsGen
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BiShengIRPythonSources.Dialects
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AnnotationOps.td
  SOURCES_GLOB dialects/annotation.py
  DIALECT_NAME annotation
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT BiShengIRPythonSources.Dialects
  ROOT_DIR "${BISHENGIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/HACCOps.td
  SOURCES_GLOB dialects/hacc.py
  DIALECT_NAME hacc
)

################################################################################
# Extensions
################################################################################

set(PYTHON_SOURCE_DIR "${BISHENGIR_SRC_DIR}/lib/Bindings/Python")

declare_mlir_python_extension(BiShengIRPythonExtensions.Main
  MODULE_NAME _bishengirRegisterEverything
  ADD_TO_PARENT BiShengIRPythonExtensions
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    RegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    BiShengIRCAPIRegisterEverything
    MLIRCAPIHIVM
    MLIRCAPIHFusion
    MLIRCAPIAnnotation
    MLIRCAPIMesh
)

declare_mlir_python_extension(BiShengIRPythonExtensions.Dialects.HIVM.Pybind
  MODULE_NAME _bishengirDialectsHIVM
  ADD_TO_PARENT BiShengIRPythonSources.Dialects.hivm
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectHIVM.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIIR
  MLIRCAPIHIVM
)

declare_mlir_python_extension(BiShengIRPythonExtensions.HIVMDialectPasses
  MODULE_NAME _bishengirHIVMPasses
  ADD_TO_PARENT BiShengIRPythonSources.Dialects.hivm
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    HIVMPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIHIVM
)

declare_mlir_python_extension(BiShengIRPythonExtensions.HFusionDialectPasses
  MODULE_NAME _bishengirHFusionPasses
  ADD_TO_PARENT BiShengIRPythonSources.Dialects.hfusion
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    HFusionPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIHFusion
)

declare_mlir_python_extension(BiShengIRPythonExtensions.AnnotationDialectPasses
  MODULE_NAME _bishengirAnnotationPasses
  ADD_TO_PARENT BiShengIRPythonSources.Dialects.annotation
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    AnnotationPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
  MLIRCAPIAnnotation
)

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  # TODO: Reduce dependencies. We need ExecutionEngine and a bunch of passes
  # for the reference backend, but logically they can be separate. But seemingly
  # the only way to handle that is to create a separate mlir python package
  # tree, which seems excessive.
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.RegisterEverything
  BiShengIRPythonSources
  BiShengIRPythonExtensions
  BiShengIRSiteInitialize
)

add_mlir_python_common_capi_library(BiShengIRAggregateCAPI
  INSTALL_COMPONENT BiShengIRPythonModules
  INSTALL_DESTINATION python_packages/bishengir/bishengir/_mlir_libs
  OUTPUT_DIRECTORY "${BISHENGIR_PYTHON_PACKAGES_DIR}/bishengir/bishengir/_mlir_libs"
  RELATIVE_INSTALL_ROOT ".."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(BiShengIRPythonModules
  ROOT_PREFIX "${BISHENGIR_PYTHON_PACKAGES_DIR}/bishengir/bishengir"
  INSTALL_PREFIX "python_packages/bishengir/bishengir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
  BiShengIRAggregateCAPI
  )
