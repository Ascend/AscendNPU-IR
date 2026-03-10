//===- HFusionStructuredOps.td - HFusion library ops -------*- tablegen -*-===//
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
// This is the operation definition file for structured operations on buffers
// that correspond to underlying library calls.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD

include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionTraits.td"
include "bishengir/Interfaces/AggregatedOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

// Base Tablegen class for HFusion ops that follows LinalgOp Interface.
class HFusionStructuredBase_Op<string mnemonic, list<Trait> props = []>
  : Op<HFusion_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"mlir::linalg::YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       DestinationStyleOpInterface,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<mlir::linalg::IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return llvm::cast<mlir::linalg::LinalgOp>(
               getOperation()).reifyResultShapes(b, reifiedReturnShapes);
    }
  }];
}

//===----------------------------------------------------------------------===//
// Named HFusion ops, implemented as a declarative configurations of generic ops.
//===----------------------------------------------------------------------===//

include "bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yamlgen.td"

//===----------------------------------------------------------------------===//
// ReduceWithIndexOp
//===----------------------------------------------------------------------===//

def ReduceWithIndexOp : HFusionStructuredBase_Op<"reduce_with_index",
    [AttrSizedOperandSegments, ResultOnlyIfTensor]> {
  let summary = "max/min reduce with index.";
  let description = [{
    Using max/min to perform a reduce operation on an AnyShaped.
    It supports two modes: (1) take input (AnyShaped) and
    index (AnyShaped), produce resulting input and index;
    (2) take input, produce resulting input and index.
    Currently only one reduction dimension is supported.
    `tie_break_left` shows whether find the leftmost index or
    the rightmost index. 
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$inits,
                       HFusion_ReduceWithIndexOpAttr:$reduce_kind,
                       BoolAttr:$tie_break_left,
                       ConfinedAttr<DenseI64ArrayAttr,
                       [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions
  );
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
      (ins
        "TypeRange":$types,
        "ValueRange":$inputs,
        "ValueRange":$inits,
        "hfusion::ReduceWithIndexKindAttr":$reduce_kind,
        "BoolAttr":$tie_break_left,
        "DenseI64ArrayAttr":$dimensions)
    >,
    OpBuilder<
      (ins
        "TypeRange":$types,
        "ValueRange":$inputs,
        "ValueRange":$inits,
        "hfusion::ReduceWithIndexKindAttr":$reduce_kind,
        "BoolAttr":$tie_break_left,
        "ArrayRef<int64_t>":$dimensions)
    >
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
#else
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              mlir::function_ref<mlir::InFlightDiagnostic()>)>
#endif
    getRegionBuilder();
  }];
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

def ArangeOp : HFusionStructuredBase_Op<"arange", [AttrSizedOperandSegments,
                                                   ResultOnlyIfTensor]> {
  let description = [{
    Differs from the classical definition of arange slightly with the addition
    of an offset(default is 0) and ability to be multi-dimensional (which also
    introduces strides). The offset and stride definition are similar to that of
    the memref descriptor.

    Given a 3D arange op, the value at each position will be:
    `arange[i, j, k] = offset + stride[0] * i + stride[1] * j + stride[2] * k`
  }];
  let arguments = (ins Optional<Index>:$offset,
                       Variadic<Index>:$strides,
                       AnyShaped:$init
  );
  let results = (outs Optional<AnyShaped>:$result_tensor);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder< (ins "Value":$init)>,
    OpBuilder< (ins "ValueRange":$strides, "Value":$init)>,
    OpBuilder< (ins "Value":$offset, "ValueRange":$strides, "Value":$init)>
  ];
  let extraClassDeclaration = structuredOpsBaseDecls # [{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
#else
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              mlir::function_ref<mlir::InFlightDiagnostic()>)>
#endif
    getRegionBuilder();
    /// Precondition: `val` must be of type ShapedType
    static void getStridesFromValue(OpBuilder & builder, Location loc,
                                    Value val,
                                    SmallVectorImpl<Value> & strides);
    static void getOffsetFromValue(OpBuilder & builder, Location loc,
                                   Value & offset);
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def GatherOp
    : HFusionStructuredBase_Op<
          "gather", [AllRanksMatch<["src", "index", "init"]>,
                     AllShapesMatch<["index", "init"]>,
                     AllElementTypesMatch<["src", "init"]>,
                     AttrWithinRankOf<"axis", "init">,
                     ResultOnlyIfTensor,
                     BiShengIRAggregatedOpInterface]>
{

  let description = [{
    Gathers one axis of the src tensor into a different with the same shape in
    all but the gather axis. Corresponds to triton.language.gather.

    Given src:tensor<16x16> and index:tensor<16x4> with axis = 1, the op is
    equivalent to:
    ```
    for i in 0 to 16 {
      for j in 0 to 4 {       // Can be tiled without consequence
        for k in 0 to 16 {    // Cannot be tiled without result potentially
                              //   becoming partial, define as gather axis
          output[i][j] = (index[i][j] == k) ? src[i][k] : output[i][j];
        }
      }
    }
    ```
    }];
  let arguments = (ins AnyShaped:$src,
                       AnyShaped:$index,
                       AnyShaped:$init,
                       I64Attr:$axis);
  let results = (outs Variadic<AnyTensor>:$result);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
                    (ins "Value":$src,
                         "Value":$indices,
                         "Value":$init,
                         "int64_t":$gather_axis)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = structuredOpsBaseDecls#[{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Get the attribute name for gather axis
    static inline StringRef getGatherAxisAttrName() {
      // Since there is only one builtin attribute, return first result
      return getAttributeNames()[0];
    }

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
#else
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              mlir::function_ref<mlir::InFlightDiagnostic()>)>
#endif
    getRegionBuilder();

    // Used for AggregateOpInterface to decompose into legal operations
    FailureOr<SmallVector<Value>> decomposeOperation(OpBuilder &);
  }];
}


//===----------------------------------------------------------------------===//
// GatherMaskOp 
//===----------------------------------------------------------------------===//
def GatherMaskOp
    : HFusionStructuredBase_Op<
          "gather_mask", [
              AllRanksMatch<["src", "mask"]>,    
              AllShapesMatch<["src", "mask"]>,        
              ResultOnlyIfTensor
          ]>
{
  let summary = [{
    Gather data from src tensor according to the non-zero values of mask tensor
  }];

  let description = [{
    Gathers elements from the `src` tensor where the corresponding element in the `mask` tensor is non-zero (True for bool type, 1 for int8 type).
    Writes the result to the pre-allocated `init` tensor (equivalent to HIVM's `dst`), 
    and returns `init` as `result` for functional compatibility.
    The `init`/`result` tensor has the same shape and element type as the `src` tensor.
    Additionally, returns `dst_size` which represents the actual length of the gathered data.

    Usage:
      result, dst_size = gather_mask(src, mask, init)  
  }];

  let arguments = (ins 
    AnyShaped:$src,
    AnyShaped:$mask,
    Variadic<AnyShaped>:$init  
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let regions = (region AnyRegion:$region);  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$mask, "ValueRange":$init)>  
  ];
  let hasCustomAssemblyFormat = 1; 
  let hasVerifier = 1;  
  let extraClassDeclaration = structuredOpsBaseDecls#[{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();

    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
#else
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              mlir::function_ref<mlir::InFlightDiagnostic()>)>
#endif
    getRegionBuilder();
  }];
}

//===----------------------------------------------------------------------===//
// Conv1DOp
//===----------------------------------------------------------------------===//

def Conv1DOp
    : HFusionStructuredBase_Op<
          "conv1d", [AllElementTypesMatch<["input","weight","bias", "init"]>,
                    ResultOnlyIfTensor,
                    BiShengIRAggregatedOpInterface]>
{

  let description = [{
    1D convolution operator with semantics aligned to 
    triton.language.extra.cann.extension.conv1d(
        input, weight, bias, groups, padding_size
    ),
    and default parameters matching 
    torch.nn.functional.conv1d(
        stride=1, dilation=1, padding_mode='zeros'
    ).

    Dimension constraints:
    - input: [iC, iW] (input channel count, input width) 
      / [N, iC, iW] (batch size, input channel count, input width)
    - weight: [oC, iC/groups, wW] (output channel count, 
      input channel count per group, convolution kernel width)
    - bias (optional): [oC] (output channel bias)
    - init/result: [oC, oW] (output channel count, output width) 
      / [N, oC, oW] (batch size, output channel count, output width)

    Configurable parameters (with default values):
    - padding: Number of zero-padding elements on both sides, default=0
    - groups: Number of groups for grouped convolution, default=1
    }];
  let arguments = (ins AnyShaped:$input,
                      AnyShaped:$weight,
                      Optional<AnyShaped>:$bias,
                      AnyShaped:$init,
                      I32Attr:$stride,
                      I32Attr:$padding,
                      I32Attr:$dilation,
                      I32Attr:$groups);
  let results = (outs AnyTensor:$result);
  let regions = (region AnyRegion:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
                    (ins "ValueRange":$inputs,
                        "Value":$output,
                        "int32_t":$stride,
                        "int32_t":$padding,
                        "int32_t":$dilation,
                        "int32_t":$groups)>];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = structuredOpsBaseDecls#[{
    // Declare functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable();
    // Declare functions necessary for LinalgStructuredInterface.
    SmallVector<utils::IteratorType> getIteratorTypesArray();
    ArrayAttr getIndexingMaps();
#ifndef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>)>
#else
    static std::function<void(mlir::ImplicitLocOpBuilder &, mlir::Block &,
                              mlir::ArrayRef<mlir::NamedAttribute>,
                              mlir::function_ref<mlir::InFlightDiagnostic()>)>
#endif
    getRegionBuilder();
  }];
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONSTRUCTUREDOPS_TD
