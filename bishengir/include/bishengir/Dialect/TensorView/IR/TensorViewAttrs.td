//===-- TensorViewAttrs.td - TensorView attributes ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_ATTRS
#define TENSORVIEW_ATTRS

include "bishengir/Dialect/TensorView/IR/TensorViewDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Encoding attributes
//===----------------------------------------------------------------------===//

class TensorView_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TensorView_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::tv";
}

def TV_PaddingValue : I32EnumAttr<
    "PaddingValue", "TensorView out-of-bounds padding value",
    [
      I32EnumAttrCase<"ZERO", 0, "zero">,
      I32EnumAttrCase<"NAN_VALUE", 1, "nan">,
      I32EnumAttrCase<"POS_INF", 2, "inf">,
      I32EnumAttrCase<"NEG_INF", 3, "-inf">
    ]> {
  let cppNamespace = "::mlir::tv";
}

// #tv.partition_view<tile=[...], dim_map=[...], padding_value=zero>
def TV_PartitionViewAttr : TensorView_Attr<"PartitionView", "partition_view"> {
  let summary = "Partition (tiling) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap,
    EnumParameter<TV_PaddingValue>:$paddingValue);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::mlir::tv::PaddingValue":$paddingValue), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap, paddingValue);
    }]>,
  ];
}

// #tv.strided_view<tile=[...], dim_map=[...], traversal_strides=[...],
//                  padding_value=zero>
def TV_StridedViewAttr : TensorView_Attr<"StridedView", "strided_view"> {
  let summary = "Strided (partition + inter-tile stride) view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$dimMap,
    ArrayRefParameter<"int64_t">:$traversalStrides,
    EnumParameter<TV_PaddingValue>:$paddingValue);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilder<(ins "::llvm::ArrayRef<int64_t>":$tile,
                     "::llvm::ArrayRef<int64_t>":$traversalStrides,
                     "::mlir::tv::PaddingValue":$paddingValue), [{
      ::llvm::SmallVector<int64_t> dimMap;
      for (int64_t i = 0, e = tile.size(); i < e; ++i) dimMap.push_back(i);
      return $_get($_ctxt, tile, dimMap, traversalStrides, paddingValue);
    }]>,
  ];
}

// #tv.gather_scatter_view<tile=[...], sparse_dim=[...], padding_value=zero>
def TV_GatherScatterViewAttr : TensorView_Attr<"GatherScatterView", "gather_scatter_view"> {
  let summary = "Gather/scatter view encoding.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$tile,
    ArrayRefParameter<"int64_t">:$sparseDim,
    EnumParameter<TV_PaddingValue>:$paddingValue);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // TENSORVIEW_ATTRS
