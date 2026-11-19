//===- VectorizeTransformOps.td - Vectorizer transforms ---*- tablegen-*-===//
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

#ifndef HIVM_VECTORIZE_TRANSFORMOPS
#define HIVM_VECTORIZE_TRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HIVMIncludeOp : Op<Transform_Dialect, "hivm.include",
                       [DeclareOpInterfaceMethods<TransformOpInterface>,
                        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Invoke a named transform sequence.";

  let arguments = (ins SymbolRefAttr:$target,
      Variadic<Transform_AnyHandleOrParamType>:$operands);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);

  let assemblyFormat = [{
    $target `failures` `(` `propagate` `)` `(` $operands `)` attr-dict `:`
        functional-type($operands, $results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyCall();
  }];
}

def HIVMPartitionOp
    : Op<Transform_Dialect, "hivm.partition",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Partition the payload root into groups.";

  let arguments = (ins TransformHandleTypeInterface:$root,
      TransformParamTypeInterface:$options, StrArrayAttr:$models);
  let results = (outs TransformParamTypeInterface:$groups);

  let assemblyFormat = [{
    $root `,` $options attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def HIVMAgglomerateOp
    : Op<Transform_Dialect, "hivm.agglomerate",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Agglomerate partition groups into a fusion plan.";

  let arguments = (ins TransformHandleTypeInterface:$root,
      TransformParamTypeInterface:$options, TransformParamTypeInterface:$groups,
      StrAttr:$selector, StrArrayAttr:$cost);
  let results = (outs TransformParamTypeInterface:$plan);

  let assemblyFormat = [{
    $root `,` $options `,` $groups attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def HIVMExtractGroupsOp
    : Op<Transform_Dialect, "hivm.extract_groups",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Extract the selected groups from a fusion plan.";

  let arguments = (ins TransformParamTypeInterface:$plan);
  let results = (outs TransformParamTypeInterface:$groups);

  let assemblyFormat = [{
    $plan attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMOutlineOp : Op<Transform_Dialect, "hivm.outline",
                       [DeclareOpInterfaceMethods<TransformOpInterface>,
                        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Outline groups into private functions plus calls.";

  let arguments = (ins TransformHandleTypeInterface:$root,
      TransformParamTypeInterface:$groups);
  let results = (outs TransformHandleTypeInterface:$newRoot,
      TransformHandleTypeInterface:$funcs);

  let assemblyFormat = [{
    $root `,` $groups attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMAnnotateVFOp
    : Op<Transform_Dialect, "hivm.annotate_vf",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Mark functions as HIVM vector functions.";

  let arguments = (ins TransformHandleTypeInterface:$funcs);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $funcs attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMLiftToLinalgOp
    : Op<Transform_Dialect, "hivm.lift_to_linalg",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Lift HIVM vector functions to linalg.";

  let arguments = (ins TransformHandleTypeInterface:$vfs);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $vfs attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMVectorizeOp
    : Op<Transform_Dialect, "hivm.vectorize",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tile and vectorize linalg operations in vector functions.";

  let arguments = (ins TransformHandleTypeInterface:$vfs,
      DefaultValuedAttr<I64Attr, "256">:$vector_length,
      DefaultValuedAttr<I64Attr, "-1">:$max_vectorize_axes);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $vfs attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMCleanupOp : Op<Transform_Dialect, "hivm.cleanup",
                       [DeclareOpInterfaceMethods<TransformOpInterface>,
                        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Clean up vector function bodies.";

  let arguments = (ins TransformHandleTypeInterface:$vfs);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $vfs attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMCheckResultOp
    : Op<Transform_Dialect, "hivm.check_result",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Verify the vector function result contract.";

  let arguments = (ins TransformHandleTypeInterface:$vfs);
  let results = (outs TransformParamTypeInterface:$result);

  let assemblyFormat = [{
    $vfs attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMTransactionOp
    : Op<Transform_Dialect, "hivm.transaction",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Submit an isolated speculative transform attempt.";

  let arguments = (ins SymbolRefAttr:$attempt,
      TransformHandleTypeInterface:$root, TransformParamTypeInterface:$options);
  let results = (outs TransformParamTypeInterface:$ticket);

  let assemblyFormat = [{
    $attempt `(` $root `,` $options `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyAttempt();
  }];
}

def HIVMTransactionAwaitOp
    : Op<Transform_Dialect, "hivm.transaction.await",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Wait for a transaction ticket and return its result record.";

  let arguments = (ins TransformParamTypeInterface:$ticket);
  let results = (outs TransformParamTypeInterface:$result);

  let assemblyFormat = [{
    $ticket attr-dict `:` functional-type(operands, results)
  }];
}

def HIVMTransactionSelectOp
    : Op<Transform_Dialect, "hivm.transaction.select",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Rank awaited candidates and publish the best one.";

  let arguments = (ins SymbolRefAttr:$cost, TransformHandleTypeInterface:$root,
      Variadic<TransformParamTypeInterface>:$candidates);
  let results = (outs TransformHandleTypeInterface:$newRoot,
      TransformParamTypeInterface:$selection);

  let assemblyFormat = [{
    $cost `(` $root (`,` $candidates^)? `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyCost();
  }];
}

def HIVMTransactionExecuteOp
    : Op<Transform_Dialect, "hivm.transaction.execute",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Internal bridge from interpreter failures to outcomes.";

  let arguments = (ins SymbolRefAttr:$attempt,
      TransformHandleTypeInterface:$root, TransformParamTypeInterface:$options);
  let results = (outs TransformParamTypeInterface:$result);

  let assemblyFormat = [{
    $attempt `(` $root `,` $options `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verifyAttempt();
  }];
}

#endif
