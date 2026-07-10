//===- AscendDPXBase.td - Ascend SIMT dialect base support -*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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