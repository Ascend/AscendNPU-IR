# BiShengIR Project.
cmake_minimum_required(VERSION 3.28.0)
project(bishengir)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(BISHENGIR_ENABLE_WERROR)
  # Treat all warnings as errors, except using deprecated api.
  add_compile_options(-Werror -Wno-error=deprecated-declarations)
  if(NOT LLVM_ENABLE_ASSERTIONS)
    # Suppress unused-variable warnings when assert is disabled.
    add_compile_options(-Wno-unused-variable -Wno-unused-but-set-variable)
  endif()
endif()

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

# The BISHENGIR_PUBLISH option is used to control behaviors that we may not want
# to expose to the users. Including: - bishengir-compile option
# `-enable-lir-compile` - internal debug features
option(BISHENGIR_PUBLISH "BiShengIR is publish" OFF)
if(BISHENGIR_PUBLISH STREQUAL "OFF")
  set(BISHENGIR_ENABLE_PM_CL_OPTIONS ON)
endif()

option(BISHENGIR_DISABLE_CANN "Disable CANN dependency" OFF)

# Only enable execution engine if the native target is available.
if(${LLVM_NATIVE_ARCH} IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_ENABLE_EXECUTION_ENGINE 1)
  add_compile_definitions(BISHENGIR_ENABLE_EXECUTION_ENGINE)
else()
  set(MLIR_ENABLE_EXECUTION_ENGINE 0)
endif()

# Build the Torch-MLIR conversion when "torch-mlir" project is enabled.
if("torch-mlir" IN_LIST LLVM_EXTERNAL_PROJECTS)
  set(BISHENGIR_ENABLE_TORCH_CONVERSIONS 1)
else()
  set(BISHENGIR_ENABLE_TORCH_CONVERSIONS 0)
endif()

# Build Triton when "triton" project is enabled.
if("triton" IN_LIST LLVM_EXTERNAL_PROJECTS)
  set(BISHENGIR_ENABLE_TRITON_COMPILE 1)
else()
  set(BISHENGIR_ENABLE_TRITON_COMPILE 0)
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
  PATHS 
    $ENV{BISHENG_INSTALL_PATH}
    NO_DEFAULT_PATH)

# -------------------------------------------------------------------------------
# Setup variables and includes
# -------------------------------------------------------------------------------

set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_MAIN_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
set(MLIR_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../mlir)

set(BISHENGIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BISHENGIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BISHENGIR_MAIN_INCLUDE_DIR ${BISHENGIR_SRC_DIR}/include)
set(BISHENGIR_INCLUDE_DIR ${BISHENGIR_BINARY_DIR}/include)
set(BISHENGIR_TOOLS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0 "${BISHENGIR_SRC_DIR}/cmake/modules"
     "${LLVM_COMMON_CMAKE_UTILS}/Modules")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(TableGen)
include(AddLLVM)
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
add_subdirectory(tools/bishengir-target-spec-tblgen)
set(BISHENGIR_TARGET_SPEC_TABLEGEN_EXE
    "${BISHENGIR_TARGET_SPEC_TABLEGEN_EXE}"
    CACHE INTERNAL "")
set(BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET
    "${BISHENGIR_TARGET_SPEC_TABLEGEN_TARGET}"
    CACHE INTERNAL "")

if(BISHENGIR_ENABLE_TORCH_CONVERSIONS)
  set(TORCH_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../torch-mlir)
  set(TORCH_MLIR_INCLUDE_DIRS ${TORCH_MAIN_SRC_DIR}/include)
  set(TORCH_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/torch-mlir/include)
  include_directories(${TORCH_TABLEGEN_OUTPUT_DIR})
  include_directories(${TORCH_MLIR_INCLUDE_DIRS})
endif()

if(BISHENGIR_ENABLE_TRITON_COMPILE)
  cmake_path(ABSOLUTE_PATH LLVM_EXTERNAL_TRITON_SOURCE_DIR
             BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set(TRITON_MAIN_SRC_DIR ${LLVM_EXTERNAL_TRITON_SOURCE_DIR})
  set(TRITON_MLIR_INCLUDE_DIRS ${LLVM_EXTERNAL_TRITON_SOURCE_DIR}/include)
  set(TRITON_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/triton/include)
  # for triton
  include_directories(${TRITON_TABLEGEN_OUTPUT_DIR})
  include_directories(${TRITON_MLIR_INCLUDE_DIRS})
  # for nvidia
  include_directories(${TRITON_MAIN_SRC_DIR}/third_party)
  include_directories(${TRITON_MAIN_SRC_DIR}/third_party/nvidia/include)
  include_directories(${LLVM_BINARY_DIR}/tools/triton/third_party)
  # for proton
  include_directories(${TRITON_MAIN_SRC_DIR}/third_party/proton/Dialect/include)
  # for RegisterTritonDialects.h
  include_directories(${TRITON_MAIN_SRC_DIR}/bin)
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

# Declare target before adding `include/bishengir`
add_custom_target(bishengir-doc)
set_target_properties(bishengir-doc PROPERTIES FOLDER "BiShengIR/Docs")


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
string(TIMESTAMP BISHENGIR_BUILD_DATE "%Y-%m-%d")
message(STATUS "BISHENGIR version: ${BISHENGIR_VERSION}")
message(STATUS "BISHENGIR build date: ${BISHENGIR_BUILD_DATE}")

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/bishengir/Version/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/bishengir/Version/Version.inc)

add_subdirectory(tools/bishengir-hfusion-ods-gen)
add_subdirectory(lib)
add_subdirectory(include/bishengir)
add_subdirectory(lib/CAPI)

if(MLIR_INCLUDE_TESTS)
  add_definitions(-DMLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  # If parent projects want to configure where to place the python packages,
  # respect that.
  if(NOT BISHENGIR_PYTHON_PACKAGES_DIR)
    set(BISHENGIR_PYTHON_PACKAGES_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  endif()
  add_subdirectory(python)
endif()

add_subdirectory(tools)

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY include/bishengir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT bishengir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT")

  install(
    DIRECTORY ${BISHENGIR_BINARY_DIR}/include/bishengir
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

set_property(GLOBAL PROPERTY BISHENGIR_OPEN_SOURCE_LIBS
                             "@BISHENGIR_OPEN_SOURCE_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_LIBS "@BISHENGIR_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_DIALECT_OPEN_SOURCE_LIBS
                             "@BISHENGIR_DIALECT_OPEN_SOURCE_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_DIALECT_LIBS "@BISHENGIR_DIALECT_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_CONVERSION_OPEN_SOURCE_LIBS
                             "@BISHENGIR_CONVERSION_OPEN_SOURCE_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_CONVERSION_LIBS
                             "@BISHENGIR_CONVERSION_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_TRANSLATION_OPEN_SOURCE_LIBS
                             "@BISHENGIR_TRANSLATION_OPEN_SOURCE_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_TRANSLATION_LIBS
                             "@BISHENGIR_TRANSLATION_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_EXTENSION_OPEN_SOURCE_LIBS
                             "@BISHENGIR_EXTENSION_OPEN_SOURCE_LIBS@")
set_property(GLOBAL PROPERTY BISHENGIR_EXTENSION_LIBS
                             "@BISHENGIR_EXTENSION_LIBS@")
