//===- HFusionTransformOps.td - HFusion transform ops -------*- tablegen-*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS

include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

def CacheReadOp : Op<Transform_Dialect, "structured.cache_read", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

  let summary = "Create a cache read of original tensor for readers.";

  let description = [{
    Create a cache read of the values given by the `targets` op handle.
    For instance, given the input IR:

    ```mlir
    "some_op"(%a) : tensor<16x16xf16> -> ()
    ```

    If the `targets` handle points to `%a`, the IR after transformation is:

    ```mlir
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
    "some_op"(%cached) : tensor<16x16xf16> -> ()
    ```

    The `targets` op handle may be associated with one or more payload IR values,
    and cache read will be performed one by one.

    #### Return modes

    The return handle points to the defining ops of the cached values.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$targets);

  let results = (outs TransformHandleTypeInterface:$cached);

  let assemblyFormat = [{
    $targets attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

def CacheWriteOp : Op<Transform_Dialect, "structured.cache_write", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

  let summary = "Create a cache write of original tensor, before storing into tensor.";

  let description = [{
    Create a cache write of the values given by the `targets` op handle.
    The `targets` payload values should be the SSA result of linalg/hfusion operations.

    For instance, given the input IR:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    ```

    If the `targets` handle points to `%a`, the IR after transformation is:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
    ```

    If the optional attr `$cache_write_to_output_init` is set to true, the binding init operand
    of the target will be used to store the cached result. *Note that this is only valid
    if the operation does not read from the init operand.*

    ```mlir
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %a = "some_op"(%empty) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%init : tensor<16x16xf16>)
    ```

    If the optional attr `output_only` is set to true, then the cached result
    will only replace the original operand in the `func.return` op.

    Following the above example, the IR after transformation is:

    ```mlir
    %a = "some_op"(%init) : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %empty = tensor.emtpy() : tensor<16x16xf16>
    %cached = linalg.copy ins(%a : tensor<16x16xf16>) outs(%empty : tensor<16x16xf16>)
    "some_use"(%a) : tensor<16x16xf16> -> ()
    // code omitted
    func.return %cached : tensor<16x16xf16>
    ```

    The `targets` op handle may be associated with one or more payload IR values,
    and cache write will be performed one by one.

    #### Return modes

    The return handle points to the defining ops of the cached values.
    This operation only reads the `targets` handle.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$targets,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$output_only,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$cache_write_to_output_init);

  let results = (outs TransformHandleTypeInterface:$cached);

  let assemblyFormat = [{
    $targets attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// SetBufferSizeOp
//===----------------------------------------------------------------------===//

def SetBufferSizeOp : Op<Transform_Dialect, "structured.set_buffer_size",
  [FunctionalStyleTransformOpTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   TransformOpInterface]> {
  let summary = "Set the allocation size of target operands.";

  let description = [{
    Indicates that the given `target` should be set to the desired size. If
    the target is statically-shaped, then nothing will be done.

    If the `target` is a `memref.alloc` or `memref.alloca` op, it
    is replaced by 1-D contiguous memref with empty layout map of i8 element
    type. It's size is determined by `static_buffer_sizes` and `unit_mode`.

    Otherwise, the op's results are annotated by `annotation.mark` op
    with the size info. Note that only results with shaped type can be annotated.

    There are two `unit_mode`:
      - "per_byte": the memref size is equal to the provided buffer size.
      - "per_element": the memref size is equal to the provided buffer size
        multiplied by the target value's element type size.

    When `reference_type` is provided, the buffer size is multiplied by the
    target value's element type size divided by the size of the
    `reference_type` (which must be a int or float type). For example:

    ```
      static buffer sizes = [1000]
      unit mode = per_byte
      reference type = i8
      target value's element type = f32
      final buffer size = 1000 * sizeof(f32) / sizeof(i8) = 4000 bytes
    ```

    The new alloc is viewed into the original alloc's shape.

    The input `target` and `static_buffer_sizes` should have the same size.
    Each `target` op handle can be associated with more than one payload op,
    and they will all be set to the same buffer size.

    Note that if the `target` already has a static shape, this transform op is nop.

    #### Return modes
    This operation consumes the target handle.
  }];

  let arguments = (ins Variadic<TransformHandleTypeInterface>:$target,
                       DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_buffer_sizes,
                       SetBufferSizeModeEnum:$unit_mode,
                       OptionalAttr<TypeAttr>:$reference_type);

  let assemblyFormat = [{
    $target
    (`static_buffer_sizes` `=` $static_buffer_sizes^)?
    `unit_mode` `=` $unit_mode
    (`reference_type` `=` $reference_type^)?
    attr-dict
    `:` type($target)
  }];

  let extraClassDeclaration = [{
     ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
   }];
}

//===----------------------------------------------------------------------===//
// HFusion MultiBufferOp
//===----------------------------------------------------------------------===//

def MultiBufferOp : Op<Transform_Dialect, "structured.multi_buffer",
  [FunctionalStyleTransformOpTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   TransformOpInterface]> {
  let summary = "mark operand to do multi buffer optimization in HFusion";

  let description = [{
    Indicates that the given `target` should be marked to do multi buffer optimization in HFusion.
    Use annotation::MarkOp to mark and set attr "hfusion.multi_buffer" with value `factor`.
    The factor means number of buffers to be enabled, and it is optional and default as 2.

    The payload op of target handle should be defined and have results, otherwise the target handle
    would not be marked. One handle may point to multiple payload ops.

    Example of payload op with one result:
    ```mlir
    %0 = linalg.exp ins(%arg0: tensor<4x8x16xf32>) outs(%arg1: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    ```

    If the above linalg.exp is matched to do multi buffer:
    ```mlir
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.multi_buffer %0 : !transform.any_op, !transform.any_op
    ```

    After transformation:
    ```mlir
    %0 = linalg.exp ins(%arg0 : tensor<4x8x16xf32>) outs(%arg1 : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
    annotation.mark %0 {hfusion.multi_buffer = 2 : i32} : tensor<4x8x16xf32>
    ```

    Example of payload op with two result:
    ```mlir
    %0, %1 = "test.source"() : () ->  (memref<1xf32>, memref<2xf32>)
    ```

    If the above test.source is matched to do multi buffer, then after transformation:
    ```mlir
    %0:2 = "test.source"() : () -> (memref<1xf32>, memref<2xf32>)
    annotation.mark %0#0 {hfusion.multi_buffer = 2 : i32} : memref<1xf32>
    annotation.mark %0#1 {hfusion.multi_buffer = 2 : i32} : memref<2xf32>
    ```

    Example using factor:
    ```mlir
    %0 = transform.structured.match ops{["linalg.exp"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.multi_buffer %0 factor = 3 : !transform.any_op
    ```

    #### Return modes
    This operation only Reads the target handle.
  }];

  let arguments = (ins Variadic<TransformHandleTypeInterface>:$target,
                       DefaultValuedOptionalAttr<I32Attr, "2">:$factor);

  let assemblyFormat = [{
    $target (`factor` `=` $factor^)? attr-dict `:` type($target)
  }];

  let extraClassDeclaration = [{
     ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
   }];
}


#endif // HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS
