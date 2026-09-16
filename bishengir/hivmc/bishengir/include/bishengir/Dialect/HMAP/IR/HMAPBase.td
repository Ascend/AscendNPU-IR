//===- HMAPBase.td - Base of hybrid mesh aware parallelism dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef HMAP_IR_HMAPBASE
#define HMAP_IR_HMAPBASE

include "mlir/IR/OpBase.td"

def HMAP_Dialect : Dialect {
  let name = "hmap";

  let cppNamespace = "::mlir::hmap";

  let description = [{
    HMAP (Hybrid Mesh Aware Parallelism) dialect.
  }];

  let dependentDialects = [
    "mesh::MeshDialect",
  ];
}

#endif // HMAP_IR_HMAPBASE
