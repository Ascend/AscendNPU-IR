//===-- TensorViewOps.td - TensorView operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_OPS
#define TENSORVIEW_OPS

include "bishengir/Dialect/TensorView/IR/TensorViewDialect.td"
include "bishengir/Dialect/TensorView/IR/TensorViewTypes.td"
include "bishengir/Dialect/TensorView/IR/TensorViewAttrs.td"
include "bishengir/Dialect/TensorView/IR/TensorViewInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

class TensorView_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorView_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// make_tensor_view : materialize a base tensor_view from a pointer.
//===----------------------------------------------------------------------===//
def TV_MakeTensorViewOp : TensorView_Op<"make_tensor_view",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "Materialize a base tensor_view (no encoding) from a pointer.";
  let arguments = (ins
    AnyTypeOf<[TV_PtrType, I64, AnyMemRef]>:$source,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `,` `sizes` `=` `[` $sizes `]` `,` `strides` `=` `[` $strides `]`
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view :
// attach a view encoding to a base tensor_view.
//===----------------------------------------------------------------------===//
class TV_MakeViewOp<string mnemonic>
    : TensorView_Op<mnemonic, [Pure, TensorViewEncodingOpInterface]> {
  let arguments = (ins TV_TensorViewType:$source);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
}

def TV_MakePartitionViewOp     : TV_MakeViewOp<"make_partition_view"> {
  let summary = "Attach a #tv.partition_view encoding to a base view.";
}
def TV_MakeStridedViewOp       : TV_MakeViewOp<"make_strided_view"> {
  let summary = "Attach a #tv.strided_view encoding to a base view.";
}
def TV_MakeGatherScatterViewOp : TV_MakeViewOp<"make_gather_scatter_view"> {
  let summary = "Attach a #tv.gather_scatter_view encoding to a base view.";
}

//===----------------------------------------------------------------------===//
// view_load / view_store : tensor and buffer access to an encoded view.
//===----------------------------------------------------------------------===//
def TV_ViewLoadOp : TensorView_Op<"view_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DestinationStyleOpInterface]> {
  let summary = "Load a view tile into a tensor or buffer, including padding.";
  let arguments = (ins TV_TensorViewType:$view, TV_TensorOrMemRef:$dst,
                       Variadic<TV_ViewIndex>:$indices);
  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` `outs` `(` $dst `:` type($dst) `)` attr-dict
    `:` qualified(type($view)) `,` type($indices)
    (`->` type($result)^)?
  }];
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

def TV_ViewStoreOp : TensorView_Op<"view_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Store the valid part of a tensor or buffer tile to a view.";
  let arguments = (ins TV_TensorViewType:$view, TV_TensorOrMemRef:$value,
                       Variadic<TV_ViewIndex>:$indices);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` `,` $value attr-dict
    `:` qualified(type($view)) `,` type($value) `,` type($indices)
  }];
}

#endif // TENSORVIEW_OPS
