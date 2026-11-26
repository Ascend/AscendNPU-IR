//===- BMUBase.td - BMU dialect base support ------------------*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
//
// Base definitions for the BMU (Buffer Manage Unit) dialect.
//
// BMU is an A6 feature: it models the A6 buffer-management hardware, and the
// llvm.hivm intrinsics it lowers to are declared under the A6 feature macro
// BSPUB_DAVINCI_BISHENGIR_A6 (llvm/include/llvm/IR/IntrinsicsHIVM.td).
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BMU_IR_BMUBASE_TD
#define BISHENGIR_DIALECT_BMU_IR_BMUBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// BMU Dialect.
//===----------------------------------------------------------------------===//

def BMU_Dialect : Dialect {
  let name = "bmu";
  let description = [{
    BMU (Buffer Manage Unit) dialect. Provides hardware-level buffer
    allocation/free and segment register access for on-chip buffers.

    A6 only: BMU is an A6 hardware unit, so this dialect and its lowering are
    only usable with an A6-enabled LLVM build (BSPUB_DAVINCI_BISHENGIR_A6).
  }];
  let cppNamespace = "::mlir::bmu";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect",
                           "::mlir::hivm::HIVMDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// BMU buffer handle type.
//===----------------------------------------------------------------------===//

// A handle on an allocation: a ranked memref whose memory space is the HIVM
// address space of the buffer (`#hivm.address_space<ub>` for UB, and so on), so
// that the allocation is usable by memref and HIVM operations; or an i64 scalar
// address for BT — BT is addressed by a value (proto "ULi") and has no address
// space. The per-op verifier checks that the handle form matches the buffer.
def BMU_BufferHandle : AnyTypeOf<[AnyMemRef, I64]>;

//===----------------------------------------------------------------------===//
// BMU base operation classes.
//===----------------------------------------------------------------------===//

class BMU_Op<string mnemonic, list<Trait> traits = []>
    : Op<BMU_Dialect, mnemonic,
          !listconcat([DeclareOpInterfaceMethods<MemoryEffectsOpInterface>],
                      traits)> {
  let hasVerifier = 1;
}

#endif // BISHENGIR_DIALECT_BMU_IR_BMUBASE_TD
