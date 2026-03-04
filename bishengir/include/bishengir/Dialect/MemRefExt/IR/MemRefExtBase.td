//====- MemRefExtBase.td - Base definitions for Extended memref dialect --====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_MEMREFEXT_BASE
#define BISHENGIR_MEMREFEXT_BASE

include "mlir/IR/OpBase.td"

def MemRefExt_Dialect : Dialect {
  let name = "memref_ext";
  let cppNamespace = "::bishengir::memref_ext";
  let description = [{
    extended memref dialect
  }];
  let dependentDialects = ["arith::ArithDialect"];
  let hasConstantMaterializer = 1;
}

#endif // BISHENGIR_MEMREFEXT_BASE