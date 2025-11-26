//===- bishengir-config.h - BiShengIR configuration --------------*- C -*-===*//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

/* This file enumerates variables from the BiShengIR configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the bishengir-c headers. */

#ifndef BISHENGIR_CONFIG_H
#define BISHENGIR_CONFIG_H

/* If set, enable conversion and compile from Torch Dialect. */
#cmakedefine01 BISHENGIR_ENABLE_TORCH_CONVERSIONS

/* If set, enables BiShengIR pass manager command line options to MLIR. */
#cmakedefine01 BISHENGIR_ENABLE_PM_CL_OPTIONS

/* If set, disable features that are currently unpublished. */
#cmakedefine01 BISHENGIR_PUBLISH

/* If set, enable BiShengIR CPU Runner. */
#cmakedefine01 MLIR_ENABLE_EXECUTION_ENGINE

/* If set, only build IR definitions. */
#cmakedefine01 BISHENGIR_BUILD_STANDALONE_IR_ONLY

/* If set, enable A5 unpublished features. */
#cmakedefine01 BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES

/* Specifies BiShengIR vendor information. */
#cmakedefine BISHENGIR_VENDOR "${BISHENGIR_VENDOR}"

/* Specifies BiShengIR repository address. */
#cmakedefine BISHENGIR_REPOSITORY "${BISHENGIR_REPOSITORY}"

/* Specifies BiShengIR build mode. */
#if defined(__GNUC__)
/* GCC and GCC-compatible compilers define __OPTIMIZE__ when optimizations are 
   enabled. */
# if defined(__OPTIMIZE__)
#  define BISHENGIR_IS_DEBUG_BUILD 0
# else
#  define BISHENGIR_IS_DEBUG_BUILD 1
# endif
#elif defined(_MSC_VER)
/* MSVC doesn't have a predefined macro indicating if optimizations are enabled.
   Use _DEBUG instead. This macro actually corresponds to the choice between
   debug and release CRTs, but it is a reasonable proxy. */
# if defined(_DEBUG)
#  define BISHENGIR_IS_DEBUG_BUILD 1
# else
#  define BISHENGIR_IS_DEBUG_BUILD 0
# endif
#else
/* Otherwise, for an unknown compiler, assume this is an optimized build. */
# define BISHENGIR_IS_DEBUG_BUILD 0
#endif

#endif
