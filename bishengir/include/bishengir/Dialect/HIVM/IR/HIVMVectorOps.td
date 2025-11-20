//===- HIVMVectorOps.td - HIVM dialect vector op defs. -----*- tablegen -*-===//
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
//
// This is the definition file for HIVM dialect vector operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMVECTOROPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMVECTOROPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

//===----------------------------------------------------------------------===//
// HIVM Vector Operations.
//===----------------------------------------------------------------------===//

class HIVM_VectorOp<string mnemonic, list<Trait> traits = [],
  list<Trait> vecSpecialTraits=[OpPipeTrait<"PIPE::PIPE_V">, VectorCoreTypeTrait]> :
  HIVM_StructuredOp<mnemonic, !listconcat(!listconcat(
    [AlwaysSpeculatable, SinglePipeOpTrait
    ], vecSpecialTraits), traits)> {
  code vectorOpBaseDecls = [{
    ::mlir::LogicalResult fold(FoldAdaptor adaptor,
        SmallVectorImpl<OpFoldResult> &results) {
      return memref::foldMemRefCast(*this);
    }

    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Elementwise N-ary Vector Ops
//===----------------------------------------------------------------------===//

class HIVM_ElementwiseNaryOp<string mnemonic, list<Trait> traits = [], Type srcType = AnyType>
    : HIVM_VectorOp<mnemonic, !listconcat([AttrSizedOperandSegments,
                                           HIVMOpSameOperandsAndResultRank,
                                           UniformReassociationFlattenTrait,
                                           CollapsibleConsecutiveTargetDimsTrait,
                                           TransposableOTF], traits)> {
  string baseClassDescription = [{
    *From the Elementwise Nary Vector Op template:*

    This operation performs element-wise operation on N operands and produces a single result.
    It may perform either transpose or broadcast along the way (but not both).

    Common constraints:
      1. Follows DestinationStyleOpInterface.
      2. The number of input operands is N; the number of output/result is one.
      3. The input/init operands and result have the same rank.
      4. The first input is vector-only.
  }];
  dag extraArguments = (ins);
  let arguments = !con((ins Variadic<srcType>:$src,
                       Variadic<AnyShaped>:$dst), extraArguments, (ins
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  ));
  let results = (outs Variadic<AnyRankedTensor>:$result);
  string extraAssemblyFormat = "";
  let assemblyFormat = [{
      attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
      (`outs` `(` $dst^  `:` type($dst) `)`)?
      }] # extraAssemblyFormat # [{
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
  }];
  code elementwiseDeclaration = [{
    /// Elementwise reassociation group is not included in redistribute like because it has transpose
    FailureOr<mlir::hivm::FlattenResult> getFlattened(mlir::hivm::FlattenOptions &options) {
      return mlir::hivm::detail::getFlattenedElementwise(*this, options);
    }
    SmallVector<int64_t> getLimitedAxes() {
      return mlir::hivm::detail::computeElementwiseLimitation(*this);
    }
    void adjustTargetDimensions(OpBuilder &builder, const FlattenResult &result) {
      return mlir::hivm::detail::adjustElementwiseTargetDimensions(builder, *this, result);
    }
    ArrayAttr getIndexingMaps() {
      return ::mlir::hivm::detail::getIndexingMapsElementwiseImpl(*this);
    }
  }];
  code elemwiseOpBaseDecls = vectorOpBaseDecls
                             # opNameDeclaration
                             # elementwiseDeclaration;
}

//===----------------------------------------------------------------------===//
// HIVM Elementwise Unary Vector Ops
//===----------------------------------------------------------------------===//

class HIVM_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = [], Type srcType = AnyType> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<1>], traits), srcType>;

def VExpOp : HIVM_ElementwiseUnaryOp<"vexp",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0, 1], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Exponential Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VAbsOp : HIVM_ElementwiseUnaryOp<"vabs",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32, I16, I32, I64]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Absolute Value Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VLnOp : HIVM_ElementwiseUnaryOp<"vln",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Natural Logarithm Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VReluOp : HIVM_ElementwiseUnaryOp<"vrelu",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32, I32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Rectified Linear Unit Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VRsqrtOp : HIVM_ElementwiseUnaryOp<"vrsqrt",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Reciprocal Square Root Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VSqrtOp : HIVM_ElementwiseUnaryOp<"vsqrt",
    [SameOperandsElementType,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Square Root Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VTanhOp : HIVM_ElementwiseUnaryOp<"vtanh",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
    ]> {
  let summary = "Elementwise Vector Hyperbolic Tangent Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VSinOp : HIVM_ElementwiseUnaryOp<"vsin",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
    ]> {
  let summary = "Elementwise Vector Sine Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VCosOp : HIVM_ElementwiseUnaryOp<"vcos",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
    ]> {
  let summary = "Elementwise Vector Cosine Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VErfOp : HIVM_ElementwiseUnaryOp<"verf",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
    ]> {
  let summary = "Elementwise Vector Error function Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VRecOp : HIVM_ElementwiseUnaryOp<"vrec",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Reciprocal Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VNotOp : HIVM_ElementwiseUnaryOp<"vnot",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Not Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VCastOp : HIVM_ElementwiseUnaryOp<"vcast",
    [VectorOnlyTrait<0>, HIVMOpSameOperandsAndResultRank,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Type Conversion Op";
  let description = baseClassDescription # [{
     Additional constraints:
       1. Supports the following conversions:

        | src  | dst  | roundingmode                                      |
        |------|------|---------------------------------------------------|
        | f32  | f32  | round, rint, floor, ceil, trunc                   |
        | f32  | f16  | round, rint, floor, ceil, trunc, odd              |
        | f32  | i64  | round, rint, floor, ceil, trunc                   |
        | f32  | i32  | round, rint, floor, ceil, trunc                   |
        | f32  | i16  | round, rint, floor, ceil, trunc                   |
        | f32  | s64  | round, rint, floor, ceil, trunc                   |
        | f32  | bf16 | round, rint, floor, ceil, trunc                   |
        | f16  | f32  | rint                                              |
        | f16  | i32  | round, rint, floor, ceil, trunc                   |
        | f16  | i16  | round, rint, floor, ceil, trunc                   |
        | f16  | i8   | round, rint, floor, ceil, trunc                   |
        | f16  | ui8  | round, rint, floor, ceil, trunc                   |
        | f16  | i4   | round, rint, floor, ceil, trunc                   |
        | bf16 | f32  | rint                                              |
        | bf16 | i32  | round, rint, floor, ceil, trunc                   |
        | ui8  | f16  | rint                                              |
        | i8   | f16  | rint                                              |
        | i8   | i1   | rint                                              |
        | i16  | f16  | round, rint, floor, ceil, trunc                   |
        | i16  | f32  | rint                                              |
        | i32  | f32  | round, rint, floor, ceil, trunc                   |
        | i32  | i64  | rint                                              |
        | i32  | i16  | rint                                              |
        | i64  | i32  | rint                                              |
        | i64  | f32  | round, rint, floor, ceil, trunc                   |
        | i4   | f16  | rint                                              |
        | i1   | f16  | rint                                              |
        | i1   | f32  | rint                                              |
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer,
                            DefaultValuedAttr<HIVM_RoundModeAttr, "RoundMode::RINT">:$round_mode
  );
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "ValueRange":$dst,
                   "hivm::RoundModeAttr":$round_mode,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>,
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "ValueRange":$dst,
                   CArg<"hivm::RoundMode", "RoundMode::RINT">:$round_mode,
                   CArg<"ArrayRef<int64_t> ", "{}">:$transpose,
                   CArg<"ArrayRef<int64_t>", "{}">:$broadcast)>
  ];
  let extraAssemblyFormat = [{
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`round_mode` `=` $round_mode^)?
    }];
  let hasVerifier = 1; // verify parameter according to hardware version
  let extraClassDeclaration = elemwiseOpBaseDecls # [{
    // Get the cast name based on src and dst types.
    std::string getCastName(bool withMode);

    Value getSingleSrc() { return getSrc()[0]; }
    Value getSingleDst() { return getDst()[0]; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Elementwise Binary Vector Ops
//===----------------------------------------------------------------------===//

class HIVM_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = [], Type srcType = AnyType> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<2>], traits), srcType>;

def VAddOp : HIVM_ElementwiseBinaryOp<"vadd",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Addition Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VMulOp : HIVM_ElementwiseBinaryOp<"vmul",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Multiplication Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VMulExtOp : HIVM_ElementwiseBinaryOp<"vmulext",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I32]>,
     VectorOnlyTrait<0>,
    DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = [{
    Elementwise Binary Vector Multiplication that Calculates
    the Most Significant 32-bits.
  }];
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support Vector-Vector operation.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VSubOp
    : HIVM_ElementwiseBinaryOp<
          "vsub", [SameOperandsElementType,
                   OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
                   DeclareOpInterfaceMethods<ExtraBufferOpInterface,
                     ["getExtraBufferSize"]>,
                   DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
                   BroadcastableOTF]> {
  let summary = "Elementwise Binary Vector Subtraction Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer);
  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "ValueRange":$src,
                   "ValueRange":$dst,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>
  ];
  let extraAssemblyFormat = [{
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VDivOp : HIVM_ElementwiseBinaryOp<
                 "vdiv", [SameOperandsElementType,
                          OperElemTypeConstraints<[0, 1], [F16, F32, I64]>,
                          DeclareOpInterfaceMethods<ExtraBufferOpInterface,
                            ["getExtraBufferSize"]>,
                          DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
                          BroadcastableOTF]> {
  let summary = "Elementwise Binary Vector Division Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support only Vector-Vector operation.
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer);
  let builders = [
    OpBuilder<(ins "TypeRange":$result,
                   "ValueRange":$src,
                   "ValueRange":$dst,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>
  ];
  let extraAssemblyFormat = [{
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VMaxOp : HIVM_ElementwiseBinaryOp<"vmax",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Maximum Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VMinOp : HIVM_ElementwiseBinaryOp<"vmin",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Minimum Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VOrOp : HIVM_ElementwiseBinaryOp<"vor",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>, CommutativeOpTrait,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Or Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support only Vector-Vector operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VAndOp : HIVM_ElementwiseBinaryOp<"vand",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>, CommutativeOpTrait,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector And Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support only Vector-Vector operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VXorOp : HIVM_ElementwiseBinaryOp<"vxor",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface, ["getExtraBufferSize"]>
    ], AnyShaped> {
  let summary = "Elementwise Binary Vector Xor Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support only Vector-Vector operation.
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer);
  let extraAssemblyFormat = [{
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "ValueRange":$dst,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>
  ];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VModOp : HIVM_ElementwiseBinaryOp<"vmod",
    [SameOperandsElementType, 
     VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Elementwise Vector Mod Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VShLOp : HIVM_ElementwiseBinaryOp<"vshl",
   [SameOperandsElementType,
    OperElemTypeConstraints<[0, 1], [I16, I32, I64]>,
    VectorOnlyTrait<0>, ScalarOnlyHWTrait<1>,
    DeclareOpInterfaceMethods<ExtraBufferOpInterface,
      ["getExtraBufferSize"]>,
    DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
    BroadcastableOTF
   ]> {
  let summary = "Elementwise Binary Vector Shift Left Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input vector and result have the same element type.
      2. Support only Vector - Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`broadcast` `=` $broadcast^)?
      (`transpose` `=` $transpose^)?
      (`->` type($result)^)?
    }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VShROp : HIVM_ElementwiseBinaryOp<"vshr",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I16, I32, I64]>,
     VectorOnlyTrait<0>, ScalarOnlyHWTrait<1>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Shift Right Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input vector and result have the same element type.
      2. Support only Vector - Scalar operation.
      3. If `round` is set to true, rounding is applied during arithmetic
         shift right.
  }];
  let builders = [OpBuilder<(ins "TypeRange":$result, "ValueRange":$src,
      "ValueRange":$dst, "BoolAttr":$round,
      CArg<"DenseI64ArrayAttr", "nullptr">:$transpose,
      CArg<"DenseI64ArrayAttr", "nullptr">:$broadcast)>];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer,
                            DefaultValuedOptionalAttr<BoolAttr, "true">:$round
  );
  let extraAssemblyFormat = [{
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`round` `:` $round^ )?
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VCmpOp : HIVM_ElementwiseBinaryOp<"vcmp",
    [OperElemTypeConstraints<[0, 1], [F16, F32, I16, I32, I64]>,
     OperElemTypeConstraints<[/*dstIdx=*/2], [I1, I8]>,
     VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Elementwise Binary Vector Comparison Op";
  let description = baseClassDescription # [{
    Compare elements from two source vector. If the comparison result is true,
    the corresponding bit of `dst` is 1 or 8.

    Additional constraints:
      1. The input vectors and output vector must have the same ranks
      2. The element type of `dst` must be bool
      3. The input is vector-only.
      4. Supports the following data type:

        |    compare mode   |       element type      |
        |-------------------|-------------------------|
        | GE/GT/LE/LT/NE/EQ | f16, f32, i16, i32, i64 |
  }];
  let extraArguments = (ins DefaultValuedAttr<HIVM_CmpModeAttr, "CompareMode::EQ">:$compare_mode);
  let extraAssemblyFormat = [{
    (`compare_mode` `=` $compare_mode^)?
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

def VPowOp : HIVM_ElementwiseBinaryOp<"vpow",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>
    ]> {
  let summary = "Elementwise Binary Vector Power Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer);
  let extraAssemblyFormat = [{
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result,"ValueRange":$src,"ValueRange":$dst)>
  ];
  let hasCanonicalizer = 1;
  let extraClassDeclaration = elemwiseOpBaseDecls;
}

//===----------------------------------------------------------------------===//
// HIVM Elementwise Ternary Vector Ops
//===----------------------------------------------------------------------===//

class HIVM_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<3>], traits)>;

def VSelOp : HIVM_ElementwiseTernaryOp<"vsel",
    [OperElemTypeConstraints<[/*condition=*/0], [I1,I8]>,
     OperElemTypeConstraints<[/*src0=*/1, /*src0=*/2], [AnyI16, F16, BF16, AnyI32, F32, I64]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface, ["getExtraBufferSize"]>,
     BroadcastableOTF]> {
    // add traits for type constraint
  let summary = "Elementwise Vector Selection Op";
  let description = baseClassDescription # [{
    Select elements from two source vector according to the binary `condition` vector.
    If the corresponding bit of the indicator is 1, select `src0`. Otherwise,
    select `src1`.

    Additional constraints:
      1. The input vectors and output vector must have the same ranks.
      2. The element type of indicator vector must be bool.
  }];
  let extraArguments = (ins Optional<AnyMemRef>:$temp_buffer);
  let extraAssemblyFormat = [{
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
  }];
  let extraClassDeclaration = elemwiseOpBaseDecls ;
}

//===----------------------------------------------------------------------===//
// HIVM Vector Broadcast Op
//===----------------------------------------------------------------------===//

def VBrcOp : HIVM_VectorOp<"vbrc",
    [SameOperandsElementType,
     OperElemTypeConstraints<
         [0], [I8, UI8, I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64, I1]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
                               ["getIteratorTypesArray", "getIndexingMaps"]>,
     DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     CollapsibleConsecutiveTargetDimsTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
    ], []> {
  let summary = "Vector Broadcast Op";
  let description = [{
    Broadcast a vector or a scalar according to the broadcast axes array.

    Constraints:
      1. The input vector and output vector must have same rank
         and the same element type.
      2. For the input operand, the size of the broadcasted axis must be 1.
      3. The broadcast indices array cannot be empty for vector input.
      4. The broadcast indices array *must* be empty for scalar input.
      5. The broadcast indices array can not be larger than the ranks of the input vector.
      6. The broadcast indices must be in `[0, RankOfSrcVec)`.
      7. For i1 type, need to ensure that the tail axis of dst is aligned with 16,
         otherwise there will be a risk of memory stampede

    Examples:
    ```mlir
    // Scalar broadcast
    hivm.hir.vbrc ins(%src : i32) outs(%dst : memref<?xi32>)
    // Vector broadcast
    hivm.hir.vbrc ins(%src : memref<1xi32>) outs(%dst : memref<?xi32>) broadcast_dims = [0]
    %result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>
    ```
  }];
  let arguments = (ins AnyType:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast_dims
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`broadcast_dims` `=` $broadcast_dims^)?
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$dst,
                   "DenseI64ArrayAttr":$broadcast_dims)>,
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$dst)>
  ];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls#[{
    static StringRef getOpName() { return "broadcast"; }
    PIPE getPipe();
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Reduce Op
//===----------------------------------------------------------------------===//

def VReduceOp : HIVM_VectorOp<"vreduce",
    [AttrSizedOperandSegments,
     OperElemTypeConstraints<[0, 1], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray", "getIndexingMaps"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     CollapsibleConsecutiveTargetDimsTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation", "getDecomposePhase"]>
    ]> {
  let summary = "Vector Reduction Op";
  let description = [{
    Recuce one or more axes of the source vector according to
    the reduction axes array, starting from an init value.

    Constraints:
      1. The input vector and output vector must have the same rank
         and the same element type.
      2. For the output operand, the size of the reduced axis must be 1.
      3. The reduction indices array can not be empty,
         nor can be larger than the ranks of the input vector.
      4. The reduced indices must be in `[0, RankOfDstVec)`.

    Examples:
    ```mlir
    hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
    %result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       HIVM_ReduceOpAttr:$arith,
                       DenseI64ArrayAttr:$reduce_dims,
                       Optional<TensorOrMemref>:$indices
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict $arith `ins` `(` $src `:` type($src) `)`
    (`indices` `(` $indices^ `:` type($indices) `)`)?
    `outs` `(` $dst `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    `reduce_dims` `=` $reduce_dims
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src,
                   "ValueRange":$dst, "hivm::ReduceOpAttr":$arith,
                   "DenseI64ArrayAttr":$reduce_dims)>,
    OpBuilder<(ins "TypeRange":$result, "Value":$src,
                   "ValueRange":$dst, "hivm::ReduceOpAttr":$arith,
                   "DenseI64ArrayAttr":$reduce_dims, "Value":$indices)>
  ];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    // Get reduction init value.
    Attribute getInit();

    static StringRef getOpName() { return "reduce"; }

    // Get the reduced value init value.
    Value getDstValue() {
      return getDst()[0];
    }

    // Get the reduced index init value.
    Value getDstIndex() {
      assert(getDst().size() >= 2 && "there should be more than 2 operands");
      return getDst()[1];
    }

    bool useVectorCrossIntr(bool lastAxis, int rank);
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Transpose Op
//===----------------------------------------------------------------------===//

def VTransposeOp : HIVM_VectorOp<"vtranspose",
    [OperElemTypeConstraints<[0], [AnyI8, AnyI16, AnyI32, F16, BF16, F32, I64, UI64]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray", "setIteratorTypesArray"]>,
     // VTranspose is considered as uniform because it is swapping two dimensions
     // and the number of min swap always exactly one.
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
    ]> {
  let summary = "Vector Transpose Op";
  let description = [{
    Permutes the dimensions of 'src' according to the given `permutation`. In
    other words:
      `dim(dst, i) = dim(src, permutation[i])`.

    Constraints:
      1. The input vector and output vector must have same rank, and the same element type.

    Examples:
    ```mlir
     hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
     %result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$permutation
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$dst,
                   CArg<"DenseI64ArrayAttr", "nullptr">:$permutation)>
  ];
  let assemblyFormat = [{
      attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`permutation` `=` $permutation^)?
      (`->` type($result)^)?
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "transpose"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Arange Op
//===----------------------------------------------------------------------===//

def VArangeOp
    : HIVM_VectorOp<
          "varange", [AttrSizedOperandSegments,
                      OperElemTypeConstraints<[0], [I16, I32, F16, F32, I64]>,
                      DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
                                                ["getIteratorTypesArray"]>,
                      // Will automatically use the default
                      // uniform reassociation impl
                      DeclareOpInterfaceMethods<FlattenInterface,
                                                ["getLimitedAxes"]>]> {
  let summary = "Vector Arange Op";
  let description = [{
    Fill a vector with range 0,1,2... based on strides and offset.
    e.g. offset = 1, strides = [1, 2], tensor/memref shape = [2x4xi32],
    the result is [[1, 3, 5, 7,
                    2, 4, 6, 8]].

    Constraints:
      1. Must have at least one stride.
      2. Default offset is 0.

    Examples:
    ```mlir
    hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
    %result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>)
                                -> tensor<32xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$dst,
                       Optional<Index>:$offset,
                       Variadic<Index>:$strides
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$dst)>,
    OpBuilder<(ins "TypeRange":$result, "Value":$dst, "Value":$offset)>
  ];
  let assemblyFormat = [{
    attr-dict
    (`offset` `[` $offset^ `]`)?
    `strides` `[` $strides `]`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result)^)?
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "arange"; }
    static void getStridesFromValue(OpBuilder &builder, Location loc,
                                    Value val, SmallVectorImpl<Value> &strides);
    static void getOffsetFromValue(OpBuilder &builder, Location loc,
                                    Value &offset);
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Interleave Op
//===----------------------------------------------------------------------===//

def VInterleaveOp : HIVM_VectorOp<"vinterleave",
    [SameOperandsElementType, AttrSizedOperandSegments,
     HIVMOpSameOperandsAndResultRank,
     OperElemTypeConstraints<[0],
       [I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>,
    ]> {
  let summary = "Vetor Interleave Op";
  let description = [{
    Interleaves the values of `N` tensors along their last dimension.
    All tensors must have the same shape.
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<I64Attr, "2">:$interleave_channel_nums
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `interleave_channel_nums` `=` $interleave_channel_nums
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "Value":$dst)>,
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "Value":$dst,
                   "int64_t":$interleave_channel_nums)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "interleave"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Deinterleave Op
//===----------------------------------------------------------------------===//

def VDeinterleaveOp : HIVM_VectorOp<"vdeinterleave",
    [SameOperandsElementType, HIVMOpSameOperandsAndResultRank,
     OperElemTypeConstraints<[0],
       [I8, I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>,
    ]> {
  let summary = "Vector Deinterleave Op";
  let description = [{
    Deinterleave one tensor along the last dimension.
    The tensor's last dimension size must be multiple of `channel_num`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       DefaultValuedOptionalAttr<I64Attr, "2">:$channel_num,
                       DefaultValuedOptionalAttr<HIVM_DeinterleaveModeAttr,
                         "DeinterleaveMode::ALL_CHANNELS">:$index_mode
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`channel_num` `=` $channel_num^)?
    (`index_mode` `=` $index_mode^)?
    (`->` type($result)^)?
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "deinterleave"; }
    int64_t getDeInterLeaveChannelNum() { return this->getChannelNum(); }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Flip Op
//===----------------------------------------------------------------------===//

def VFlipOp : HIVM_VectorOp<"vflip",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0, 1], [I8, UI8, I16, I32, UI16, UI32, I64, UI64, F16, F32, BF16]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
                               ["getLimitedAxes"]>
    ]> {
  let summary = "Vector Flip Op";
  let description = [{
    Flips a tensor along the last dimension.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result)^)?
  }];
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "flip"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector MulExtended Op
//===----------------------------------------------------------------------===//

def VMulextendedOp : HIVM_VectorOp<"vmulextended",
    [AttrSizedOperandSegments, HIVMOpSameOperandsAndResultRank,
     OperElemTypeConstraints<[0], [I16]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray",
       ]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
                               ["getLimitedAxes"]>
     ]> {
  let summary = "Vector Mulextended Op";
  let description = [{
    Do vmul on two tensors. Get both high and low 16-bits.
  }];
  let arguments = (ins Variadic<TensorOrMemref>:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "ValueRange":$src, "ValueRange":$dst)>
  ];
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "mul_extended"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Pad Op
//===----------------------------------------------------------------------===//

def VPadOp : HIVM_VectorOp<"vpad",
    [HIVMOpSameOperandsAndResultRank,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["adjustTargetDimensions", "getLimitedAxes"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation", "getDecomposePhase"]>]> {
  let summary = "Vector Pad Op";
  let description = [{
    Pads the input operand. Operation semantic is similar to
    `tensor.pad`.

    Arguments:
      * `src`: the tensor/memref on which to pad values
      * `dst`: reserved for bufferization
      * `pad_value`: the value to pad
      * `low`: the padding lengths along the start of each dimension
      * `high`: the padding lengths along the end of each dimension

    Example:
    ```mlir
    hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>)
                  low[%first_dim_low, 0] high[%first_dim_high, 0]
                  pad_value %pad_value : f32
                    -> tensor<?x16xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       AnyType:$pad_value,
                       Variadic<Index>:$low,
                       Variadic<Index>:$high,
                       DenseI64ArrayAttr:$static_low,
                       DenseI64ArrayAttr:$static_high
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `low` `` custom<DynamicIndexList>($low, $static_low)
    `high` `` custom<DynamicIndexList>($high, $static_high)
    `pad_value` $pad_value `:` type($pad_value)
    (`->` type($result)^)?
  }];
  let hasCanonicalizer = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "pad"; }
    SmallVector<OpFoldResult>
    getMixedPadImpl(ArrayRef<int64_t> staticAttrs, ValueRange values);
    SmallVector<OpFoldResult> getMixedLowPad();
    SmallVector<OpFoldResult> getMixedHighPad();
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Concat Op
//===---------------------------------------------------------------------q-===//

def VConcatOp : HIVM_VectorOp<"vconcat",
    [SameOperandsElementType,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["adjustTargetDimensions", "getLimitedAxes"]>
    ]> {
  let summary = "Vector Concatenation Op";
  let description = [{
    The concat operation constructs a tensor out of a variadic list of input
    tensors, concatenated along a static dimension number. All inputs and the
    result type must share the same rank.

    `dim` specifies the dimension along which to concatenate. The size of the
    concatenated dimension in the result must be equal to the sum of the sizes
    of the inputs along that dimension. All other dimensions in both the inputs
    and result must be the same size.

    Example:
    ```mlir
    hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>)
                            outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
    ```
  }];
  let arguments = (ins I64Attr:$dim,
                       Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    `dim` `(` $dim `)` attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result)^)?
  }];
  let extraClassDeclaration = vectorOpBaseDecls;
}

//===----------------------------------------------------------------------===//
// HIVM Gather Op
//===----------------------------------------------------------------------===//

def VGatherOp : HIVM_VectorOp<"vgather",
    [HIVMOpSameOperandsAndResultRank,
     OperElemTypeConstraints<[0], [I16, UI16, I32, UI32, F16, BF16, F32]>,
     OperElemTypeConstraints<[1], [I32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     // Will automatically use the default uniform reassociation impl
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>
    ]> {
  let summary = "Vector Gather Op";
  let description = [{
    Retrieve elements from a tensor/memref according to given indices, 
    and store these elements in another tensor/memref.
    The gather axis is the last dimension.

    Arguments:
      * `src`: the tensor/memref from which to gather elements
      * `indices`: the indices of elements to gather from src
      * `dst`: the tensor/memref to store elements
      * `temp_buffer`: extra memory required by gather op
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$indices,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `indices` `(` $indices `:` type($indices) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src, "Value":$indices, "Value":$dst)>
  ];
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "gather"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Cumprod Op
//===----------------------------------------------------------------------===//

def VCumprodOp : HIVM_VectorOp<"vcumprod",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0], [I16, I32, F16, F32, I64]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Vector Cumprod Op";
  let description = [{
    Calculate the cumulative product of each element along the specified axis 
    of `src`. Each element along the specified axis in the output of cumprod 
    contains the product of all elements from the first element to the current 
    position in the original `src`.

    Constraints:
      1. The input vector and output vector must have the same rank
         and the same element type.

    Arguments:
      * `src`: the tensor/memref from which to calculate the cumulative sum
      * `dst`: the tensor/memref to store elements
      * `cum_dims`: specifies the dimension along which to calculate the 
                    cumulative product.

    Examples:
    ```mlir
    hivm.hir.vcumprod ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0]
    %result = hivm.hir.vcumprod ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] -> tensor<?xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       ConfinedAttr<DenseI64ArrayAttr,
                         [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `cum_dims` `=` $cum_dims
    (`->` type($result)^)?
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "cumprod"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Cumsum Op
//===----------------------------------------------------------------------===//

def VCumsumOp : HIVM_VectorOp<"vcumsum",
    [SameOperandsElementType,
     OperElemTypeConstraints<[0], [I16, I32, F16, F32, I64]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Vector Cumsum Op";
  let description = [{
    Calculate the cumulative sum of each element along the specified axis of 
    `src`. Each element along the specified axis in the output of cumsum 
    contains the sum of all elements from the first element to the current 
    position in the original `src`.

    Constraints:
      1. The input vector and output vector must have the same rank
         and the same element type.

    Arguments:
      * `src`: the tensor/memref from which to calculate the cumulative sum
      * `dst`: the tensor/memref to store elements
      * `cum_dims`: specifies the dimension along which to calculate the
                    cumulative sum.

    Examples:
    ```mlir
    hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0]
    %result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] -> tensor<?xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       ConfinedAttr<DenseI64ArrayAttr,
                         [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `cum_dims` `=` $cum_dims
    (`->` type($result)^)?
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "cumsum"; }
  }];
}

//===----------------------------------------------------------------------===//
// HIVM Vector Sort Op
//===----------------------------------------------------------------------===//

def VSortOp : HIVM_VectorOp<"vsort",
    [AttrSizedOperandSegments,
     OperElemTypeConstraints<[0, 1], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
    ]> {
  let summary = "Vector Sort Op";
  let description = [{
    Sort the sorting axis of `src` in ascending or descending order, and output
    the sorted value and the index corresponding to the value.

    Constraints:
      1. The input vector and output vector must have the same rank.
      2. Currently only tail axis sorting is supported.

    Arguments:
      * `src`: the tensor/memref from which to be sorted
      * `dst_value`: the tensor/memref to store the sorted value
      * `dst_index`: the tensor/memref to store the index corresponding to dst_value
      * `descending`: determines whether to sort in ascending or descending
                      order. The default is false, which means ascending order
      * `sort_axis`: Axis to be sorted

    Examples:
    ```mlir
    hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
    %result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$descending,
                       DefaultValuedAttr<I64Attr, "-1">:$sort_axis
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `descending` `=` $descending
    `sort_axis` `=` $sort_axis
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`->` type($result)^)?
  }];
  let builders = [
    OpBuilder<(ins "TypeRange":$result, "Value":$src,
                   "ValueRange":$dst, "bool":$descending, "int64_t":$sort_axis)>];
  let hasVerifier = 1;
  let extraClassDeclaration = vectorOpBaseDecls # [{
    static StringRef getOpName() { return "sort"; }

    // Get the sort value.
    Value getDstValue();

    // Get the sort index.
    Value getDstIndex();

    // Get Signed Sort Axis
    int64_t getSignedSortAxis();
  }];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMVECTOROPS_TD
