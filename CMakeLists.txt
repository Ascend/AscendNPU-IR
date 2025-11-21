# BiShengIR Project.
cmake_minimum_required(VERSION 3.28.0)
project(bishengir)

# bishengir-target-spec-tblgen related target has ninja version requirement
if(CMAKE_GENERATOR MATCHES "Ninja")
  execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} --version
    OUTPUT_VARIABLE NINJA_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(NINJA_VERSION ${NINJA_VERSION} CACHE STRING "Ninja version number" FORCE)
  if("${NINJA_VERSION}" VERSION_LESS "1.12.0")
    message(FATAL_ERROR "Ninja version too old, need at least 1.12.0")
  endif()
endif()

# -------------------------------------------------------------------------------
# Handle Option
# -------------------------------------------------------------------------------

# The BISHENGIR_PUBLISH option is used to control behaviors that is unpublished.
option(BISHENGIR_PUBLISH "BiShengIR is publish" OFF)

option(BISHENGIR_BUILD_EXAMPLES
  "Build the BiShengIR example programs. If OFF, just generate build targets." OFF)

# Build the Torch-MLIR conversion when "torch-mlir" project is enabled.
if("torch-mlir" IN_LIST LLVM_EXTERNAL_PROJECTS)
  set(BISHENGIR_ENABLE_TORCH_CONVERSIONS 1)
else()
  set(BISHENGIR_ENABLE_TORCH_CONVERSIONS 0)
endif()

# The BISHENGIR_BUILD_STANDALONE_IR_ONLY option is used to only build IR definitions.
option(BISHENGIR_BUILD_STANDALONE_IR_ONLY
  "Experimental feature. If set, only build IR definitions." OFF)

# Only enable execution engine if the native target is available.
if (${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()

if(BISHENGIR_ENABLE_TORCH_CONVERSIONS AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR "torch-mlir can't build with sharded libs.")
endif()

set(LLVM_LIT_ARGS
    "-sv"
    CACHE STRING "lit default options")

find_program(
  BISHENG_COMPILER_EXECUTABLE
  NAMES bisheng
  PATHS $ENV{PATH})

# -------------------------------------------------------------------------------
# Setup variables and includes
# -------------------------------------------------------------------------------

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
set(MLIR_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(BISHENGIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bishengir)
set(BISHENGIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BISHENGIR_MAIN_INCLUDE_DIR ${BISHENGIR_SRC_DIR}/include)
set(BISHENGIR_INCLUDE_DIR ${BISHENGIR_BINARY_DIR}/bishengir/include)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${BISHENGIR_SRC_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(AddLLVM)
include(TableGen)
include(AddMLIR)
include(AddMLIRPython)
include(AddBiShengIR)

configure_file(
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Config/bishengir-config.h.cmake
  ${BISHENGIR_INCLUDE_DIR}/bishengir/Config/bishengir-config.h)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_MAIN_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${BISHENGIR_MAIN_INCLUDE_DIR})
include_directories(${BISHENGIR_INCLUDE_DIR})

# Adding tblgen tools here as calling add_tablegen sets some variables in
# PARENT_SCOPE which gets lost if that folder is included from another directory
# like tools
add_subdirectory(bishengir/tools/bishengir-target-spec-tblgen)
set(BISHENGIR_TARGET_SPEC_TABLEGEN_EXE
    "${BISHENGIR_TARGET_SPEC_TABLEGEN_EXE}"
    CACHE INTERNAL "")
set(BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET
    "${BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET}"
    CACHE INTERNAL "")

add_subdirectory(bishengir/tools/bishengir-options-tblgen)
set(BISHENGIR_OPTIONS_TABLEGEN_EXE
    "${BISHENGIR_OPTIONS_TABLEGEN_EXE}"
    CACHE INTERNAL "")
set(BISHENGIR_OPTIONS_TABLEGEN_TARGET
    "${BISHENGIR_OPTIONS_TABLEGEN_TARGET}"
    CACHE INTERNAL "")

if(BISHENGIR_ENABLE_TORCH_CONVERSIONS)
  set(TORCH_MAIN_SRC_DIR ${LLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR})
  set(TORCH_MLIR_INCLUDE_DIRS ${TORCH_MAIN_SRC_DIR}/include)
  set(TORCH_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/torch-mlir/include)
  include_directories(${TORCH_TABLEGEN_OUTPUT_DIR})
  include_directories(${TORCH_MLIR_INCLUDE_DIRS})
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

if(BISHENGIR_BUILD_EXAMPLES)
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH}
        CACHE STRING "Ascend CANN install path")
  if(${ASCEND_HOME_PATH} STREQUAL "")
    message(FATAL_ERROR "Ascend CANN package is required for building example!")
  endif()
  add_subdirectory(bishengir/test/Integration)
endif()

# Declare target before adding `include/bishengir`
add_custom_target(bishengir-doc)
set_target_properties(bishengir-doc PROPERTIES FOLDER "BiShengIR/Docs")

# Custom target to install all bishengir products for release
add_custom_target(bishengir-publish-products)
set_target_properties(bishengir-publish-products PROPERTIES FOLDER "BiShengIR/Tools")
add_llvm_install_targets(install-bishengir-publish-products
                         COMPONENT bishengir-publish-products)

# If BISHENGIR_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
if(NOT DEFINED BISHENGIR_VERSION_MAJOR)
  set(BISHENGIR_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()
if(NOT DEFINED BISHENGIR_VERSION_MINOR)
  set(BISHENGIR_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()
if(NOT DEFINED BISHENGIR_VERSION_PATCHLEVEL)
  set(BISHENGIR_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()
if(NOT DEFINED BISHENGIR_VERSION_SUFFIX)
  set(BISHENGIR_VERSION_SUFFIX ${LLVM_VERSION_SUFFIX})
endif()
set(BISHENGIR_VERSION "${BISHENGIR_VERSION_MAJOR}.${BISHENGIR_VERSION_MINOR}.${BISHENGIR_VERSION_PATCHLEVEL}${BISHENGIR_VERSION_SUFFIX}")
message(STATUS "BISHENGIR version: ${BISHENGIR_VERSION}")

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bishengir/include/bishengir/Version/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/bishengir/include/bishengir/Version/Version.inc)

add_subdirectory(bishengir/tools/bishengir-hfusion-ods-gen)
if (BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  add_subdirectory(bishengir/include/bishengir/Interfaces)
  set(STANDALONE_IR_BUILD_DIALECT
  Annotation
  HFusion
  HIVM
  MemRef
  MemRefExt
  Scope
  Symbol
  Tensor
  )
  foreach(d IN LISTS STANDALONE_IR_BUILD_DIALECT)
    add_subdirectory(bishengir/include/bishengir/Dialect/${d})
    add_subdirectory(bishengir/lib/Dialect/${d}/IR)
  endforeach()
  # TODO: Math and Utils should be added in STANDALONE_IR_BUILD_DIALECT
  add_subdirectory(bishengir/include/bishengir/Dialect/MathExt)
  add_subdirectory(bishengir/lib/Dialect/HIVM/Utils)
  add_subdirectory(bishengir/lib/Dialect/Math)
  add_subdirectory(bishengir/lib/Dialect/Utils)
else()
  add_subdirectory(bishengir/include/bishengir)
  add_subdirectory(bishengir/lib)
endif()
add_subdirectory(bishengir/lib/CAPI)

if(MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_subdirectory(bishengir/test)
endif()

# TODO: Standalone build should support python binding
if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY AND MLIR_ENABLE_BINDINGS_PYTHON)
  # If parent projects want to configure where to place the python packages,
  # respect that.
  if(NOT BISHENGIR_PYTHON_PACKAGES_DIR)
    set(BISHENGIR_PYTHON_PACKAGES_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/bishengir/python_packages")
  endif()
  add_subdirectory(bishengir/python)
endif()

if(NOT BISHENGIR_BUILD_STANDALONE_IR_ONLY)
  add_subdirectory(bishengir/tools)
endif()

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY bishengir/include/bishengir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT bishengir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT")

  install(
    DIRECTORY ${BISHENGIR_BINARY_DIR}/bishengir/include/bishengir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT bishengir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE)

  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-bishengir-headers DEPENDS
                             bishengir-headers COMPONENT bishengir-headers)
  endif()
endif()

set_property(GLOBAL PROPERTY BISHENGIR_LIBS "@BISHENGIR_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS "@BISHENGIR_DIALECT_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS "@BISHENGIR_CONVERSION_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_TRANSLATION_LIBS "@BISHENGIR_TRANSLATION_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_EXTENSION_LIBS "@BISHENGIR_EXTENSION_LIBS@")
