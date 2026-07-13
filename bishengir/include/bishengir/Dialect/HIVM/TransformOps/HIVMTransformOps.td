//===- HIVMTransformOps.td - HIVM transform ops -------------*- tablegen-*-===//
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

#ifndef HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS
#define HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// MapForallToHIVMBlocks
//===----------------------------------------------------------------------===//

def MapForallToHIVMBlocks :
  Op<Transform_Dialect, "hivm.map_forall_to_blocks",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{
    The operation maps each `scf.forall` given by the target handle to 
    HIVM block ops. Mapping is one-to-one and the induction
    variables of `scf.forall` are rewritten to hivm block idx ops.
    
    ### Restrictions:

    - Non-normalized `scf.forall` ops, or the ones with dynamic trip counts are
    currently not supported.
    - Only **bufferized** `scf.forall` are currently supported.
    - Only `scf.forall` distributed to **at most 1 dimension** are
    currently supported.

    #### Return modes:

    If the `target` handle points to any `scf.forall` that doesn't meet the above
    requirments, the transform definitely fails.
    The return handle points to the block idx op. This operation consumes 
    the target handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $target attr-dict `:` functional-type($target, $result)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // HIVM_TRANSFORMOPS_HIVMTRANSFORMOPS