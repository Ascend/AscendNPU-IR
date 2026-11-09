//===-- TensorViewTypes.td - TensorView types -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_TYPES
#define TENSORVIEW_TYPES

include "bishengir/Dialect/TensorView/IR/TensorViewDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class TensorView_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TensorView_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// tv.ptr : global pointer type.
//===----------------------------------------------------------------------===//
def TV_PtrType : TensorView_Type<"Ptr", "ptr"> {
  let summary = "TensorView global pointer type (`!tv.ptr<T>`).";
  let parameters = (ins "Type":$pointeeType);
  let assemblyFormat = "`<` $pointeeType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType), [{
      return $_get(pointeeType.getContext(), pointeeType);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// tv.tensor_view : structured GM view.  The optional encoding attribute selects
// the view kind (null = base, or one of partition/strided/gather_scatter).
//
//   !tv.tensor_view<1024xf32, strides=[1]>
//   !tv.tensor_view<?x?xf32, strides=[?,1], #tv.partition_view<...>>
//===----------------------------------------------------------------------===//
def TV_TensorViewType : TensorView_Type<"TensorView", "tensor_view"> {
  let summary = "TensorView structured GM view type.";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$strides,
    "Attribute":$encoding);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins
        "::llvm::ArrayRef<int64_t>":$shape, "Type":$elementType,
        "::llvm::ArrayRef<int64_t>":$strides,
        CArg<"Attribute", "{}">:$encoding), [{
      return $_get(elementType.getContext(), shape, elementType, strides, encoding);
    }]>,
  ];
  let extraClassDeclaration = [{
    // Rank of the view (number of shape dimensions).
    int64_t getRank() const { return getShape().size(); }
  }];
}

//===----------------------------------------------------------------------===//
// Derived type constraints.
//===----------------------------------------------------------------------===//
def TV_TensorOrMemRef : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;
def TV_IndexLikeTensor : RankedTensorOf<[Index, I32, I64]>;
def TV_IndexBuffer : MemRefOf<[Index, I32, I64]>;
def TV_IndexVector : AnyTypeOf<[TV_IndexLikeTensor, TV_IndexBuffer]>;
def TV_ViewIndex : AnyTypeOf<[Index, TV_IndexVector]>;

#endif // TENSORVIEW_TYPES
