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

#endif
