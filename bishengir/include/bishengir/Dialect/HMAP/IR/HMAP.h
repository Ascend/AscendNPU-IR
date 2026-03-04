//===- HMAP.h - Hybrid Mesh Aware Parallelism dialect ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HMAP_IR_HMAP_H
#define BISHENGIR_DIALECT_HMAP_IR_HMAP_H

#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// HMAP Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HMAP/IR/HMAPOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// HMAP Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HMAP/IR/HMAPOps.h.inc"

#endif // BISHENGIR_DIALECT_HMAP_IR_HMAP_H
