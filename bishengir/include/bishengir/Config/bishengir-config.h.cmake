//===- bishengir-config.h - BiShengIR configuration --------------*- C -*-===*//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* This file enumerates variables from the BiShengIR configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the bishengir-c headers. */

#ifndef BISHENGIR_CONFIG_H
#define BISHENGIR_CONFIG_H

/* If set, enable conversion and compile from Torch Dialect. */
#cmakedefine01 BISHENGIR_ENABLE_TORCH_CONVERSIONS

/* If set, enables BishengIR pass manager command line options to MLIR. */
#cmakedefine01 BISHENGIR_ENABLE_PM_CL_OPTIONS

/* If set, disable features that we don't want to expose to users. */
#cmakedefine01 BISHENGIR_PUBLISH

/* If set, enable conversion and compile from Triton Dialect. */
#cmakedefine01 BISHENGIR_ENABLE_TRITON_COMPILE

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
