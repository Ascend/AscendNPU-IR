//===- TritonExtDialect.td - TritonExt dialect definition ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TritonExt dialect for Ascend-specific extensions to
// TritonGPU shared memory layouts.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRITONEXT_DIALECT
#define BISHENGIR_TRITONEXT_DIALECT

include "mlir/IR/OpBase.td"

def TritonExt_Dialect : Dialect {
  let name = "ttgext";

  let cppNamespace = "::bishengir::triton_ext";

  let description = [{
    BishengIR extensions to the TritonGPU dialect.
    Contains Ascend-specific layout encodings that extend the core
    TritonGPU shared memory layout system.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class TritonExt_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TritonExt_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
