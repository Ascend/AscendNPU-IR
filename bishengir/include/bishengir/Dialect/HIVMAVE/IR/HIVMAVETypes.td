//===- AVEVFTypes.td - AVE regbased  types ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AVE dialect regbased types.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_VF_TYPES
#define AVE_VF_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "bishengir/Dialect/HIVMAVE/IR/HIVMAVEBase.td"

//===----------------------------------------------------------------------===//
// AVE RegBased type definitions
//===----------------------------------------------------------------------===//
class AVE_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<AVE_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def AVE_MaskType : AVE_Type<"AVE_MaskType", "mask"> {
  let summary = "AVE Hardware mask type";
  let description = [{
    In our hardware, we have logical masks and physical masks.
    For example, we have a mask 128xi1 that all true:
    Logical:
    1 1 1 1 1 1 ... 128x1
    Physical:
    1 0 1 0 1 0 ... 128x(1 0)

    we use `!ave.mask<128xi1, b16>` to express this syntax.
  }];
  let parameters = (ins "int64_t":$length,
                        "MaskWidth":$mask_width,
                        "Layout":$layout,
                        "int":$part);
  let assemblyFormat = "`<` $length `x` $mask_width `,` $layout `,` $part `>`";
let extraClassDeclaration = [{
    ArrayRef<int64_t> getShape() {
      return ArrayRef { getLength() };
    }

    uint32_t getAveMaskWidthLength() {
      return (uint32_t) getMaskWidth();
    }
  }];
}

def AVE_PaddedVectorType : AVE_Type<"PaddedVector",
                                      "pad_vec",
                                      [ShapedTypeInterface]> {
  let summary = [{
    Vector with padded layout.
  }];
  let description = [{
    In our hardware, we have fixed vector length, we call it `VL`. For the vector
    has length that less than VL, we have to pad it into VL. There are 4 padding
    stratigies: even, odd, lower, higher. Outs of padding, this type also takes
    elemtype and length as arguments.
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "Type":$elementType,
                        "Layout":$layout, "int":$part);
  let assemblyFormat = "`<` $shape`x`$elementType `,` $layout `,` $part `>`";
  let extraClassDeclaration = [{
    bool isLayoutIntlv() {
      return getLayout() == Layout::intlv;
    }

    bool isLayoutPlanar() {
      return getLayout() == Layout::planar;
    }

    bool hasRank() {
      return true;
    }

    ::mlir::hivmave::PaddedVectorType cloneWith(
      ::std::optional<ArrayRef<int64_t>> shape, ::mlir::Type elementType) {
        return ::mlir::hivmave::PaddedVectorType::get(getContext(),
                                                   shape.value_or(getShape()),
                                                   elementType,
                                                   getLayout(), getPart());
    }
  }];
  // we should garantee that out $length x $elemtype should not exceed out VL
  // length in verifier. let genVerifyDecl = 1;
}

class AveVector<list<Type> elementTypes>:
  AnyTypeOf<[ShapedContainerType<elementTypes, CPred<"IsAllowedAVEBitsVector($_self)">,
               "a 1-D or 0-D vector with <= 2048 bits ::mlir::VectorType">,
             AVE_PaddedVectorType]>;

def AveMaskType :
  AnyTypeOf<[ShapedContainerType<[I1], CPred<"IsAllowedAVEBitsVector($_self)">,
               "i1 type, less than 256 bits">,
             AVE_MaskType]>;

def AveValidVector : AveVector<
  [AnyI1, AnyI8, AnyI16, AnyI32, BF16, F16, F32, F8E4M3FN, F8E5M2]>
{
  let summary = "vector<N x i1>, vector<N x si8>, vector<N x ui8>, vector<N x si16>, "
                "vector<N x ui16>, vector<N x si32>, vector<N x ui32>, "
                "vector<N x f16>, vector<N x f32>, total bits <= 2048";
}

def AveExtValidVector : AveVector<
  [AnyI1, AnyI8, AnyI16, AnyI32, AnyI64, F8E4M3FN, F8E5M2, BF16, F16, F32]>
{
  let summary = "vector<N x i1>, vector<N x si8>, vector<N x ui8>, vector<N x si16>, "
                "vector<N x ui16>, vector<N x si32>, vector<N x ui32>, "
                "vector<N x si64>, vector<N x f8>, vector<N x bf16>, vector<N x f16>, "
                "vector<N x f32>, total bits <= 2048 (4096 if dtype == i64)";
}

class HasMatchingMaskTypeConstraint<string vector, string mask> :
  OptionalTypesMatchWith<
    mask # " has i1 element type and same shape as " # vector,
    vector, mask,
    "::llvm::cast<mlir::VectorType>($_self).cloneWith({}, IntegerType::get($_ctxt, 1))">;

#endif // AVE_VF_TYPES
