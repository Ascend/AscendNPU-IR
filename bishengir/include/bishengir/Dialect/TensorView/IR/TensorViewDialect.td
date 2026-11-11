//===-- TensorViewDialect.td - TensorView dialect def. --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_DIALECT
#define TENSORVIEW_DIALECT

include "mlir/IR/OpBase.td"

def TensorView_Dialect : Dialect {
  let name = "tv";
  let cppNamespace = "::mlir::tv";
  let summary = "The TensorView dialect: structured global-memory views.";

  let description = [{
    TensorView (`tv`) is a view-family dialect that sits between the Triton
    `tt` dialect and the HIVM / AscendNPU-IR level.  It materializes structured
    views (`tensor_view`) out of pointers (`tv.ptr`) and loads/stores tiles of
    those views as `tensor` SSA values, keeping the compute side purely in the
    tensor domain.  It is independent of and parallel to the TileIR dialect.
  }];


  // `tv.ptr` / `tv.tensor_view` and the encoding attributes rely on the default
  // mnemonic-based dispatch for their assembly.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // TENSORVIEW_DIALECT
