//===- MemRefExtOps.td - MemRefExt op definitions -----------*- tablegen -*-===//
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

#ifndef BISHENGIR_MEMREFEXT_OPS
#define BISHENGIR_MEMREFEXT_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "bishengir/Dialect/MemRefExt/IR/MemRefExtBase.td"

class MemRefExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<MemRefExt_Dialect, mnemonic, traits>;

def MemRefExt_AllocWorkspaceOp
    : MemRefExt_Op<"alloc_workspace",
                   !listconcat([AttrSizedOperandSegments], [])> {
  let summary = "workspace memory allocation operation.";
  let description = [{
    The `alloc_workspace` operation allocates a region of workspace memory.
    Example:

    ```mlir
    memref_ext.alloc_workspace() : memref<100xi8>
    memref_ext.alloc_workspace(%dynamic) : memref<2x?xi32>
    memref_ext.alloc_workspace(%dynamic) from %arg offset = [%offset] : from memref<?xi8> to memref<2x?xi32>
    ```
  }];
  let arguments = (ins Optional<AnyMemRef>:$workspaceArg,
                       Variadic<Index>:$dynamicSize,
                       Variadic<Index>:$offset);
  let results = (outs Res<AnyMemRef, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$memref);

  let extraClassDeclaration = [{
    ::mlir::MemRefType getType() {
      return ::llvm::cast<::mlir::MemRefType>(getResult().getType());
    }
  }];

  let assemblyFormat = [{
    `(`$dynamicSize`)` (`from` $workspaceArg^)? ( `offset` `=` `[`$offset^`]`)?
        attr-dict `:` (`from` type($workspaceArg)^ `to`)? type($memref)
  }];

  let hasVerifier = 1;
}

#endif // BISHENGIR_MEMREFEXT_OPS