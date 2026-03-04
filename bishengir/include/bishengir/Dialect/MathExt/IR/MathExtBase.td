//===- MathExtBase.td - Base definitions for math ext dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATHEXT_BASE
#define MATHEXT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def MathExt_Dialect : Dialect {
  let name = "mathExt";

  let cppNamespace = "::mlir::mathExt";

  let description = [{
    Extended Math dialect.
  }];
  let hasConstantMaterializer = 1;
}

#endif // MATHEXT_BASE
