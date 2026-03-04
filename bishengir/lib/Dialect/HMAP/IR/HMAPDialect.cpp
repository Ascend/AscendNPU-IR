//===- HMAPDialect.cpp - Implementation of HMAP dialect and ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HMAP/IR/HMAP.h"

using namespace mlir;
using namespace mlir::hmap;

//===----------------------------------------------------------------------===//
// HMAPDialect
//===----------------------------------------------------------------------===//

void hmap::HMAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/HMAP/IR/HMAPOps.cpp.inc"
      >();
}

#include "bishengir/Dialect/HMAP/IR/HMAPOpsDialect.cpp.inc"
