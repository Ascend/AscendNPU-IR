//===- Passes.td - Pass pass definition file ---------------*- tablegen -*-===//
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

#ifndef BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_TD
#define BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def CanonicalizeTensorReshape
    : Pass<"canonicalize-tensor-reshape", "func::FuncOp"> {
  let summary = "Canonicalize tensor reshape";
  let constructor = "mlir::tensor::createCanonicalizeTensorReshapePass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def FoldTensorEmpty : Pass<"fold-tensor-empty", "func::FuncOp"> {
let summary = "Fold tensor empty";
  let constructor = "mlir::tensor::createFoldTensorEmptyPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def NormalizeTensorOps : Pass<"normalize-tensor-ops", "func::FuncOp"> {
  let summary = "Optimization patterns for tensor ops";
  let constructor = "mlir::tensor::createNormalizeTensorOpsPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def NarrowTensorOp : Pass<"narrow-tensor-ops", "func::FuncOp"> {
let summary = "Narrow liveness of tensor ops";
  let constructor = "mlir::tensor::createNarrowTensorOpPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def PropagateReshape : Pass<"propagate-reshape", "func::FuncOp"> {
  let summary = "Propagate operations through reshape operations.";
  let description = [{
    This pass identifies reshape operations where the source is a linalg operation,
    and tries to propagate the operation through the reshape. It transforms patterns like:

    %a = elemwise %b, %c
    %reshape = reshape %a

    into:

    %reshape_b = reshape %b
    %reshape_c = reshape %c
    %a = elemwise %reshape_b, %reshape_c

    The pass then replaces all occurrences of the original %reshape with %a.
  }];
  let options = [
      Option<"forHIVM", "for-hivm", "bool",
             /*default=*/"false",
             "Do propagate reshape for HIVM, which has special behavior">
  ];
  let constructor = "mlir::tensor::createPropagateReshapePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def TrickleConcatDown : Pass<"trickle-concat-down", "func::FuncOp"> {
let summary = "Trickle tensor concat down";
  let constructor = "mlir::tensor::createTrickleConcatDownPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def BubblePadUp : Pass<"bubble-pad-up", "func::FuncOp"> {
let summary = "Bubble up tensor padOp";
  let constructor = "mlir::tensor::createBubblePadUpPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def NormalizeLastDimUnalignedTensorOp : Pass<"normalize-last-dim-unaligned-tensor-op", "func::FuncOp"> {
  let summary = "Normalize tensor ops when the last dim is unaligned.";
  let description = [{
    This pass identifies Tensor Concat operations where the concat axis is the last dimension
    and transposes the inputs and outputs such that the concat axis becomes the first dimension.

    For example:

    ```mlir
    %0 = tensor.concat dim(1) %A, %B : (tensor<16x32xf32>, tensor<16x64xf32>) ->  tensor<16x96xf32>
    ```

    Would become:

    ```mlir
    %0 = tensor.empty() : tensor<32x16xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<16x32xf32>) outs(%0 : tensor<32x16xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<64x16xf32>
    %transposed_0 = linalg.transpose ins(%arg1 : tensor<16x64xf32>) outs(%1 : tensor<64x16xf32>) permutation = [1, 0]
    %concat = tensor.concat dim(0) %transposed, %transposed_0 : (tensor<32x16xf32>, tensor<64x16xf32>) -> tensor<96x16xf32>
    %2 = tensor.empty() : tensor<16x96xf32>
    %transposed_1 = linalg.transpose ins(%concat : tensor<96x16xf32>) outs(%2 : tensor<16x96xf32>) permutation = [1, 0]
    ```

    This pass also identifies Tensor Pad operations where the last dimension is being padded and transposes
    the inputs and outputs such that the padding dimension becomes the first dimension.

    For example:

    ```mlir
    %0 = tensor.pad %A low[0, 3] high[0, 2] {...} : tensor<16x27xf32> to tensor<16x32xf32>
    ```

    Would become:

    ```mlir
    %0 = tensor.empty() : tensor<27x16xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<16x27xf32>) outs(%0 : tensor<27x16xf32>) permutation = [1, 0]
    %padded = tensor.pad %transposed low[3, 0] high[2, 0] {...} : tensor<27x16xf32> to tensor<32x16xf32>
    %1 = tensor.empty() : tensor<16x32xf32>
    %transposed_0 = linalg.transpose ins(%padded : tensor<32x16xf32>) outs(%1 : tensor<16x32xf32>) permutation = [1, 0]
    ```
  }];
  let constructor = "mlir::tensor::createNormalizeLastDimUnalignedTensorOpPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
    "linalg::LinalgDialect",
  ];
}

def BubbleUpExtractSlice : Pass<"bubble-up-extract-slice", "func::FuncOp"> {
  let summary = "Bubble up extract slice";
  let description = [{Bubble up extract slice}];
  let constructor = "::mlir::tensor::createBubbleUpExtractSlicePass()";
  let dependentDialects = [];
  let options =
      [Option<
           "aggressive", "aggressive", "bool", "false",
           "Aggressive mode will try to bubble up extract slices even if "
           "source value has more then one uses">,
  ];
}

def MergeConsecutiveInsertExtractSlice 
    : Pass<"merge-consecutive-insert-extract-slice", "func::FuncOp"> {
  let summary = "Merge consecutive insert extract slice";
  let description = [{Merge consecutive insert extract slice}];
  let constructor = "::mlir::tensor::createMergeConsecutiveInsertExtractSlicePass()";
  let dependentDialects = [];
}

def DecomposeTensorConcat : Pass<"decompose-tensor-concat", "func::FuncOp">{
  let summary = "Decompose tensor::ConcatOp using upstream patterns";
  let description = [{Decompose tensor::ConcatOp using upstream patterns}];
  let constructor = "mlir::tensor::createDecomposeTensorConcatPass()";
}

def OptimizeDpsOpWithYieldedInsertSlice :
  Pass<"optimize-dps-op-with-yielded-insert-slice", "func::FuncOp"> {
  let summary = "Optimize destination style op's that are inserted and yielded.";
  let constructor = [{
    mlir::tensor::createOptimizeDpsOpWithYieldedInsertSlicePass()
  }];
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

def ReplicateOutEmptyTensor
    : Pass<"replicate-out-empty-tensor", "func::FuncOp"> {
  let summary = "Clone tensor empty for destination style interface";
  let constructor = "mlir::tensor::createReplicateOutEmptyTensorPass()";
  let dependentDialects = [
    "tensor::TensorDialect",
  ];
}

#endif // BISHENGIR_DIALECT_TENSOR_TRANSFORMS_PASSES_TD
