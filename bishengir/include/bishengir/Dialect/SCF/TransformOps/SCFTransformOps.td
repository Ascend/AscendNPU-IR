//===- SCFTransformOps.td - SCF (loop) transformation ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_SCF_TRANSFORM_OPS
#define BISHENGIR_SCF_TRANSFORM_OPS

include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ParallelLoopTileOp
//===----------------------------------------------------------------------===//

def ParallelLoopTileOp : Op<Transform_Dialect, "parallel_loop.tile", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

  let arguments =
     (ins TransformHandleTypeInterface:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$static_sizes);

  let results = (outs Variadic<TransformHandleTypeInterface> : $loops);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter & rewriter,
        ::mlir::transform::TransformResults & transformResults,
        ::mlir::transform::TransformState & state);
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LoopTileOp
//===----------------------------------------------------------------------===//

def LoopTileOp : Op<Transform_Dialect, "loop.tile", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {
  
  // FIXME: Add description
  let description = [{
    Indicates that the given `target` loop should be tiled with the given sizes.
    This transform generates a loop nest.
    
    Currently tiling is limited to `scf.ForOp`, and the number of tile size
    can only be one (i.e., only support tiling one loop into two).

    Tile sizes may be known at transformation time, in which case they are
    expected to be provided in the `static_size` attribute, or not, in which
    case the tile value must be computed by the payload IR and the handle to the
    operation computing it must be provided through `dynamic_sizes`. When the
    sizes are not known statically, the corresponding entry in the
    `static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
    the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
    be as many handles as `ShapedType::kDynamic` values in the
    `static_sizes` attribute. A static size of `0` indicates that the dimension
    should not be tiled. No loop will be generated for such dimensions. If all
    tile sizes are `0`, this transform is effectively a no-op.

    This op returns handles to the generated loops. The number of loops is the
    number of tile sizes that are statically known to be non-zero.

    #### Return modes

    On success, the resulting handles are associated with the list of tiled loops.

    This operation only supports tiling `scf.ForOp` and produces a definite
    failure otherwise. The number of input tile sizes can only be one, otherwise
    a definite failure is produced.

    This operation produces a silenceable failure if the `dynamic_sizes` handles
    are associated with lists of payload operations of a size different than
    that of the list associated with the `target` handle.

    If the internal implementation of tiling for any of the operations fails,
    produces a definite failure.
  }];

  let arguments =
    (ins TransformHandleTypeInterface:$target,
         Variadic<Transform_AnyHandleOrParamType>:$dynamic_sizes,
         DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_sizes,
         DefaultValuedAttr<BoolAttr, "false">:$is_npart_mode,
         DefaultValuedAttr<BoolAttr, "false">:$is_reorder_mode
         );

  let results = (outs Variadic<TransformHandleTypeInterface>:$loops);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ForToForallOp
//===----------------------------------------------------------------------===//

def ForToForallOp : Op<Transform_Dialect, "loop.for_to_forall", [
  FunctionalStyleTransformOpTrait,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TransformOpInterface]> {

    let description = [{
    Indicates that the given `for_op`, which is a single `scf.for` op,
    should be converted to a `scf.forall` op.

    If non-empty, the `mapping` is added as an attribute to the
    resulting `scf.forall`.
    
    Note that the given loop can either yield no results, or only yield values
    produced by `tensor.insert_slice` ops. In that case, insert slices will be
    mapped to `tensor.parallel_insert_slice` ops.

    If `annotate_only` is true, the transform op will only add
    `map_for_to_forall` attribute to the target loop handle. If `mapping`
    is present, they will also be added as attribute.

    #### Return modes

    On success, the resulting handles are associated with the `scf.forall` op.

    This operation produces a definite failure if :
      - The number of payload ops is not one.
      - The payload op is not a `scf.for` op.
      - When present, the number of mapping attributes is not one.
      - The `scf.for` op has either no yielded results, or only yield values
        produced by `tensor.insert_slice` ops.
  }];

  let arguments = (ins TransformHandleTypeInterface:$for_op,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping,
                   DefaultValuedAttr<BoolAttr, "false">:$annotate_only);

  let results = (outs TransformHandleTypeInterface:$forallOp);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];

  let assemblyFormat = "$for_op attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// LoopNormalizeOp
//===----------------------------------------------------------------------===//

def LoopNormalizeOp : Op<Transform_Dialect, "loop.normalize",
    [FunctionalStyleTransformOpTrait, 
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface]> {
  let summary = "Normalizes the given scf.for loop";
  let description = [{
    Makes the target scf.for op has step 1 while preserving trip count.
    Currently only support normalizing loops with static and
    non-zero lower bounds.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";

  let results = (outs TransformHandleTypeInterface:$transformed);

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &transformResults,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // BISHENGIR_SCF_TRANSFORM_OPS
