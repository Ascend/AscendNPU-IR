//===-- HACC.cpp - C Interface for HACC dialect -------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir-c/Dialect/HACC.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HACC, hacc, mlir::hacc::HACCDialect)
