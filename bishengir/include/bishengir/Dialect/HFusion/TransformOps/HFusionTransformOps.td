//===- HFusionTransformOps.td - HFusion transform ops -------*- tablegen-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS
#define HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS

include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformEnums.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Linalg/TransformOps/LinalgTransformEnums.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// GetFuncArgumentOp
//===----------------------------------------------------------------------===//

def GetFuncArgumentOp : Op<Transform_Dialect, "func.get_func_argument", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

  let description = [{
    Get the arguments of the `target` function.
    The `target` op handle must be associated with exactly one payload op, and
    it must point to a `func.func` op.

    There are three possible modes:
      - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
        function arguments at the specified positions.
      - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
        all arguments except those at the given positions.
      - All, i.e. `%target[all]`. This will return all arguments of the function.

    If `find_reshape_consumer` option is set, the transform op will try to get handles to
    the function argument after being reshaped.

    #### Return modes

    This operation fails definitely if the target op handle does not point to
    a `func.func` op. It produces a silenceable failure if any of the indices
    exceeds the number of arguments of the target function.

    The return handle points to the function arguments when `find_reshape_consumer`
    is not set. Otherwise, the return handle *may* point to the function arguments
    after being reshaped, if and only if the argument is reshaped only once.
    If the argument does not satisfy this condition, a definite failure is produced.

    This operation only reads the target handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$raw_position_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all,
                       UnitAttr:$find_reshape_consumer);

  let results = (outs TransformValueHandleTypeInterface:$outputs);

  let assemblyFormat =
        "$target `[`"
        "custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)"
        "`]` attr-dict "
        "`:` custom<SemiFunctionType>(type($target), type($outputs))";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

//===----------------------------------------------------------------------===//
// GetFuncResultOp
//===----------------------------------------------------------------------===//

def GetFuncResultOp : Op<Transform_Dialect, "func.get_func_result", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

  let description = [{
    Get the result of the `target` function (a.k.a., the value returned by `func.return` op).
    The `target` op handle must be associated with exactly one payload op, and
    it must point to a `func.func` op.

    There are three possible modes:
      - Position list directly, i.e. `%target[0, 1, 2]`. This will return the
        function arguments at the specified positions.
      - Inverted position list, i.e. `%target[except(0, 1, 2)]`. This will return
        all arguments except those at the given positions.
      - All, i.e. `%target[all]`. This will return all arguments of the function.

    If `find_reshape_producer` option is set, the transform op will try to get handles to
    the value that is reshaped before returned.

    #### Return modes

    This operation fails definitely if the target op handle does not point to
    a `func.func` op. It produces a silenceable failure if any of the indices
    exceeds the number of arguments of the target function.

    The return handle points to the function arguments when `find_reshape_producer`
    is not set. Otherwise, the return handle *may* point to the value that is reshaped
    before returned, if and only if the returning value is the result of a reshape op.
    If the result does not satisfy this condition, a definite failure is produced.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$raw_position_list,
                       UnitAttr:$is_inverted,
                       UnitAttr:$is_all,
                       UnitAttr:$find_reshape_producer);

  let results = (outs TransformValueHandleTypeInterface:$outputs);

  let assemblyFormat =
        "$target `[`"
        "custom<TransformMatchDims>($raw_position_list, $is_inverted, $is_all)"
        "`]` attr-dict "
        "`:` custom<SemiFunctionType>(type($target), type($outputs))";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

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
// ReverseOp
//===----------------------------------------------------------------------===//

def ReverseOp : Op<Transform_Dialect, "reverse",
  [FunctionalStyleTransformOpTrait,
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   TransformOpInterface]> {
  let description = [{
    This transform op gets and reverses the list of operations held by the
    input `target` handle.

    This transform reads the `target` handle and produces the `result` handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
  let assemblyFormat = [{
    $target attr-dict `:` functional-type($target, $result)
  }];
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
//===----------------------------------------------------------------------===//

def ExtendedFuseIntoContainingOp :
  Op<Transform_Dialect, "structured.extended_fuse_into_containing_op",
     [DeclareOpInterfaceMethods<TransformOpInterface,
          ["allowsRepeatedHandleOperands"]>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      ReportTrackingListenerFailuresOpTrait,
      AttrSizedResultSegments,
      FunctionalStyleTransformOpTrait]> {
  let summary = "Fuse a producer into a containing operation.";

  let description = [{
    Fuses the `producer_op` into the `containing_op`.
    Returns a handle to the fused ops and the `new_containing_op`.

    The producer is typically a slice of a tileable op (i.e., implements
    TilingInterface). In that case, this transform computes the accessed
    producer slice inside of the containing op ("tile and fuse") and if required,
    creates a new containing op with outputs from the fused producer. Otherwise,
    the entire producer is cloned inside the containing op ("clone and fuse").

    Each containing op handle must be associated with exactly one payload op. The
    producer op handle may be associated with multiple payload ops. This
    transform fuses producers one-by-one, always picking an unspecified producer
    that has at least one use inside the containing op among the
    producers. A producer can be listed multiple times in the handle.

    If the `producer_op` has uses that are post-dominated by the `containing_op`,
    then it is fused into `containing_op` completely to avoid recomputation.
    This behavior can be disabled by setting `duplicate_producer` to true.

    Note: If a producer has multiple uses inside the containing op, a union
    of the requested regions is computed, and each consumer will only access the
    region it needs via slicing.

    #### Return modes

    If at least one producer could not be fused, this operation produces a
    silenceable failure.  This is the case when tiling fails or when no
    producer op could be found among the remaining producers that has at least
    one use within the containing op. I.e., "producers" that are not consumed
    within the containing op are rejected by this operation.

    This operation consumes the producer handle.
    This operation only reads the containing op handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$producer_op,
                       Variadic<TransformHandleTypeInterface>:$containing_op,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$duplicate_producer);

  let results = (outs Variadic<TransformHandleTypeInterface>:$fused_op,
                      Variadic<TransformHandleTypeInterface>:$new_containing_op);

  let builders = [
    OpBuilder<(ins "Value":$producerOp, "Value":$containingOp)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure fuseIntoOneContaining(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &results,
        ::mlir::transform::TransformState &state,
        size_t index,
        ::mlir::Operation* containingOp);
  }];

  let hasCustomAssemblyFormat = 1;
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

//===----------------------------------------------------------------------===//
// MatchAncestorOfOp
//===----------------------------------------------------------------------===//

def MatchAncestorOfOp : Op<Transform_Dialect, "structured.match.ancestor_of",
    [MemoryEffectsOpInterface,
     NavigationTransformOpTrait,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Match op that is an ancestor of the child op within the target op, while
    satisfying additional constraints.

    Note: Operations that are not within the same block as child is not
    considered as ancestor.

    The following constraints are supported:
      - interface: an optional MatchInterfaceEnum specifying an enum
        representation for an interface to target.
      - ops: an optional StrArrayAttr specifying the concrete name of an op.
        Multiple names can be specified. Matched ops must have one of specified
        names.
      - required attributes: the matched op must have all specified attributes
        (with their specified values).
      - optional attributes: the match op must have at least one of the specified
        attributes (with their specified values).
      - filter_result_type: the matched op must return exactly this one type.
      - filter_operand_types: all the operands of the matched op must must be of
        this type. If more than a type is specified, then the length of the list
        must be equal to the number of operands in the matched op, and the match
        will succeed only if the operand types match all the types in the list
        in the order in which they are specified.

    Note: Only ops that satisfy all specified constraints are matched.

    #### Return modes

    This op traverses the ops nested under `target` and returns the handles to
    all the operations that match the requirements.

    This op fails if the target is not a handle to exactly one operation.
    Otherwise it succeeds.

    This operation does not consume the target handle and produces new handles:
    it is a navigation op.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target,
                       TransformHandleTypeInterface:$child,
                       OptionalAttr<StrArrayAttr>:$ops,
                       OptionalAttr<MatchInterfaceEnum>:$interface,
                       OptionalAttr<DictionaryAttr>:$op_attrs,
                       OptionalAttr<DictionaryAttr>:$optional_op_attrs,
                       OptionalAttr<TypeAttr>:$filter_result_type,
                       OptionalAttr<TypeArrayAttr>:$filter_operand_types
  );
  let results = (outs TransformHandleTypeInterface:$results);
  let assemblyFormat = [{
    (`ops` `{` $ops^ `}`)?
    (`interface` `{` $interface^ `}`)?
    (`attributes` $op_attrs^)?
    (`optional_attributes` $optional_op_attrs^)?
    (`filter_result_type` `=` $filter_result_type^)?
    (`filter_operand_types` `=` $filter_operand_types^)?
    `in` $target
    `ancestor` `of` $child
    attr-dict
    `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$target, "Value":$child,
                   "ArrayRef<StringRef>":$opNames)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$target, "Value":$child,
                   "ArrayRef<StringRef>":$opNames)>,
    OpBuilder<(ins "Value":$target, "Value":$child,
                   "ArrayAttr":$ops, "DictionaryAttr":$op_attrs,
                   "DictionaryAttr":$optional_op_attrs)>,
    OpBuilder<(ins "Value":$target, "Value":$child,
                   "ArrayAttr":$ops, "DictionaryAttr":$op_attrs)>
  ];
}

//===----------------------------------------------------------------------===//
// ExtendedLoopOutlineOp
//===----------------------------------------------------------------------===//

def ExtendedLoopOutlineOp : Op<Transform_Dialect, "loop.extended_outline_op",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Extended Outlines loops into a single named function";
  // TODO: change this.
  let description = [{
    Moves the loops into a separate function with the specified name and replaces
    the loop in the Payload IR with a call to that function. Takes care of
    forwarding values that are used in the loop as function arguments. The 
    provided name is used as a _base_ for forming actual function names following 
    `SymbolTable` auto-renaming scheme to avoid duplicate symbols. Expects that 
    all ops in the Payload IR have a `SymbolTable` ancestor (typically true because
    of the top-level module).

    Example: 
    ```c++
    %function, %call = transform.loop.extended_outline_op 
                                 %loops_9, %loops_11, %loops_13, %loops_15
                             {func_name = "_attn_fwd_outlined_vf_4"} :
                             (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op) 
                             -> (!transform.any_op, !transform.any_op)
    ```

    #### Return Modes

    Returns a handle to the outlined function and a handle to the
    corresponding function call operations in the same order as the operand
    handle.

    Produces a definite failure if outlining failed for any of the targets.
  }];

  // Note that despite the name of the transform operation and related utility
  // functions, the actual implementation does not require the operation to be
  // a loop.
  let arguments = (ins Variadic<TransformHandleTypeInterface>:$targets,
                   StrAttr:$func_name);
  let results = (outs TransformHandleTypeInterface:$function,
                      TransformHandleTypeInterface:$call);

  let assemblyFormat =
    "$targets attr-dict `:` functional-type(operands, results)";
}

#endif // HFUSION_TRANSFORMOPS_HFUSIONTRANSFORMOPS
