//===- HACCBase.td - Base definitions for HACC dialect -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base heterogeneous async computing
// call support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD

include "mlir/IR/OpBase.td"

def HACC_Dialect : Dialect {
  let name = "hacc";
  let description = [{
    Heterogeneous Async Computing Call (HACC) dialect.
  }];
  let cppNamespace = "::mlir::hacc";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "mlir::DLTIDialect"
  ];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
