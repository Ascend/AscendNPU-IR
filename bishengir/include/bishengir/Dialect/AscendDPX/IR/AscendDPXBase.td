//===- AscendDPXBase.td - Ascend SIMT dialect base support -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPXBASE_TD
#define BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPXBASE_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// AscendDPX Dialect.
//===----------------------------------------------------------------------===//

def AscendDPX_Dialect : Dialect {
  let name = "ascend_dpx";
  let description = [{
    Ascend David SIMT virtual ISA definitions.
  }];
  let cppNamespace = "::mlir::ascend_dpx";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// AscendDPX Dialect operations.
//===----------------------------------------------------------------------===//

class AscendDPX_Op<string mnemonic, list<Trait> traits = []>
    : Op<AscendDPX_Dialect, mnemonic, traits>;

#endif // BISHENGIR_DIALECT_ASCENDDPX_IR_ASCENDDPXBASE_TD