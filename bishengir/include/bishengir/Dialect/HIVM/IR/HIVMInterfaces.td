//===- HIVMInterfaces.td - HIVM dialect interfaces defs ----*- tablegen -*-===//
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
// This is the definition file for the interface for HIVM dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "bishengir/Interfaces/AggregatedOpInterface.td"
include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.td"
include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.td"
include "bishengir/Dialect/HIVM/Interfaces/FlattenInterface.td"
include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.td"

def HIVMCoreTypeInterface : OpInterface<"CoreTypeInterface"> {
  let description = [{
     Interface to query the core type of HIVM Ops.

     There are 2 use cases:
     1. CoreType is staticly attatched to an op by trait 'CoreTypeTrait'
     2. CoreType is dynamicly inferred through 'InferCoreTypeInterface'
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the core type of an op.",
    /*retTy=*/"std::optional<::mlir::hivm::TCoreType>",
    /*methodName=*/"getCoreType",
    /*args=*/(ins),
    /*methodBody=*/[{
      return ::mlir::hivm::detail::queryCoreTypeHelper(tablegen_opaque_val);
    }]>,
  ];
}

def HIVMInferCoreTypeInterface : OpInterface<"InferCoreTypeInterface"> {
  let description = [{
     Interface to infer the core type of HIVM Ops.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the core type of an op.",
    /*retTy=*/"std::optional<::mlir::hivm::TCoreType>",
    /*methodName=*/"inferCoreType",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
  ];
}

def HIVMStructuredOpInterface : OpInterface<"HIVMStructuredOp",
    [DestinationStyleOpInterface, // for buferization
     OpPipeInterface,             // for sync analysis
     HIVMCoreTypeInterface,       // for mix kernel related analysis
     FlattenInterface             // for HIVM flattening
    ]> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    Hybrid Intelligence Virtual Machine (HIVM) Structured Operation interface.
  }];
  let methods = [
    //===------------------------------------------------------------------===//
    // Operation property handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op is an elemwise n-ary op.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isElemwiseNaryOp",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::isElemwiseNaryOpImpl($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op support inline broadcast.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInlineBroadcastable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ConcreteOp::template hasTrait<mlir::OpTrait::BroadcastableOTF>();
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/[{
        Return the broadcast attribute name.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getBroadcastAttrString",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return "broadcast";
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the list of inline-broadcasted axes for the operand.
        Directly return inline-broadcasted axes of op when set opOperand nullptr.
      }],
      /*retTy=*/"mlir::SmallVector<int64_t>",
      /*methodName=*/"getInlinedBroadcastableAxes",
      /*args=*/(ins "OpOperand* ":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return ::mlir::hivm::detail::getInlinedBroadcastableAxes($_op, opOperand);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the op support inline transpose.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isInlineTransposable",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ConcreteOp::template hasTrait<mlir::OpTrait::TransposableOTF>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether the op supports inline broadcast and has a broadcast array.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"existInlineBroadcastLoopDims",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          if (!$_op.isInlineBroadcastable())
            return false;
          return !$_op.getBroadcastArray().empty();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether the op support inline transpose and has a permutation array.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"existInlineTransposeLoopDims",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          if (!$_op.isInlineTransposable())
            return false;
          return !$_op.getPermutationArray().empty();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether the op has a hardware-unsupported scalar operand.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasHWUnsupportedScalarOperand",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          return mlir::hivm::hasHWUnsupportedScalarOperandImpl($_op);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Loop types handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the total number of loops within the current operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getIteratorTypesArray().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::count($_op.getIteratorTypesArray(),
                           mlir::hivm::IteratorType::kParallel);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are parallel loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getParallelLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kParallel, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are reduction loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getReductionLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kReduction, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are broadcast loops (including inline broadcast).
      }],
      /*retTy=*/"void",
      /*methodName=*/"getBroadcastLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kBroadcast, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are transpose loops (including inline transpose).
      }],
      /*retTy=*/"void",
      /*methodName=*/"getTransposeLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kTranspose, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the permutation array.
      }],
      /*retTy=*/"mlir::ArrayRef<int64_t>",
      /*methodName=*/"getPermutationArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getPermutationArray($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the broadcast array.
      }],
      /*retTy=*/"mlir::ArrayRef<int64_t>",
      /*methodName=*/"getBroadcastArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getBroadcastArray($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are pad loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getPadLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kPad, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are concat loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getConcatLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kConcat, res);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dims that are gather loops.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getGatherLoopDims",
      /*args=*/(ins "SmallVectorImpl<int64_t> &":$res),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::findPositionsOfType($_op.getIteratorTypesArray(),
          mlir::hivm::IteratorType::kGather, res);
      }]
    >,
    //===------------------------------------------------------------------===//
    // Other interface methods.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types of the current operation.
      }],
      /*retTy=*/"mlir::SmallVector<::mlir::hivm::IteratorType>",
      /*methodName=*/"getIteratorTypesArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.isElemwiseNaryOp())
          return hivm::detail::getIteratorTypesArrayForElemwiseOp($_op);

        llvm_unreachable("get iterator not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the iterator types of the current operation.
      }],
      /*retTy=*/"llvm::LogicalResult",
      /*methodName=*/"setIteratorTypesArray",
      /*args=*/(ins "const IteratorType":$iteratorType, "const DenseI64ArrayAttr &":$arrayAttr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.isElemwiseNaryOp())
           return hivm::detail::setIteratorTypesArrayForElemwiseOp($_op, iteratorType, arrayAttr);

        llvm_unreachable("set iterator not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get memory effects for HIVM Structured Op.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getEffects",
      /*args=*/(ins "::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<MemoryEffects::Effect>>&":$effects),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.hasPureTensorSemantics()) {
          return;
        }
        ::mlir::hivm::detail::getEffectsImpl(effects, cast<hivm::HIVMStructuredOp>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        get Operands with target memref space.
      }],
      /*retTy=*/"mlir::SmallVector<mlir::Value>",
      /*methodName=*/"getTargetSpaceOperands",
      /*args=*/(ins "hivm::AddressSpace":$addressSpace, "bool":$includeTmpBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getTargetSpaceOperandsImpl(
          $_op, addressSpace, includeTmpBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get the op with or without temp buffer
      }],
      /*retTy=*/"mlir::SmallVector<::mlir::OpOperand *>",
      /*methodName=*/"getHIVMOperands",
      /*args=*/(ins "bool":$includeExtraBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getHIVMOperandsImpl($_op, includeExtraBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get the op type with or without temp buffer
      }],
      /*retTy=*/"mlir::SmallVector<::mlir::Type>",
      /*methodName=*/"getHIVMOperandTypes",
      /*args=*/(ins "bool":$includeExtraBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getHIVMOperandTypesImpl($_op, includeExtraBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get the input operand with or without temp buffer
      }],
      /*retTy=*/"mlir::SmallVector<::mlir::OpOperand *>",
      /*methodName=*/"getHIVMInputOperands",
      /*args=*/(ins "bool":$includeExtraBuffer),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getHIVMInputOperandsImpl($_op, includeExtraBuffer);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       check if the operand is vector only at a specific index
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isVectorOnlyOperand",
      /*args=*/(ins "size_t": $idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::isVectorOnlyOperandImpl($_op, idx);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get mask contiguous dimension of this current op
      }],
      /*retTy=*/"mlir::BitVector",
      /*methodName=*/"getContiguousAxes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getContiguousAxesImpl($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get mask unit dimension of this current op
      }],
      /*retTy=*/"mlir::BitVector",
      /*methodName=*/"getUnitAxesMask",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getUnitAxesMaskImpl($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Get mask permuted dimension of this current op
      }],
      /*retTy=*/"mlir::BitVector",
      /*methodName=*/"getPermutedAxesMask",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getPermutedAxesMaskImpl($_op);
      }]
    >,
    /// Need to implement this for every concrete op
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps attribute within the current operation.
      }],
      /*retTy=*/"mlir::ArrayAttr",
      /*methodName=*/"getIndexingMaps",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getIndexingMaps not implemented");
        return ::mlir::ArrayAttr();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing maps within the current operation.
      }],
      /*retTy=*/"mlir::SmallVector<mlir::AffineMap>",
      /*methodName=*/"getIndexingMapsArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto range = $_op.getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to compute all the operand
        subshapes given loop bounds. This is used to answer the question: "given
        an iteration space over the codomain, what are the subshapes of the
        operands involved in the computation".
        The default behavior is to just concatenate all the indexing maps.
        A custom AffineMap allows providing a map that can be used to
        compute subshapes even in cases where the concatenation of indexing maps
        (i.e. the data traversal order) is not a simple permutation of the loop
        traversal order. It is then possible to define ops with skewed data
        traversal order for which we can still easily compute hyperrectangular
        loop bounds and subviews.
      }],
      /*retTy=*/"mlir::AffineMap",
      /*methodName=*/"getLoopsToShapesMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto maps = $_op.getIndexingMapsArray();
// TODO : release different version of ascendnpu ir and remove the macro
#ifndef __LLVM_MAJOR_VERSION_21_COMPATIBLE__
        return concatAffineMaps(maps);
#else
        return concatAffineMaps(maps, $_op.getContext());
#endif
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Hook to provide a custom AffineMap used to construct the
        hyperrectangular loop iteration space given all the operand subshapes.
        This is used to answer the question:
        "Given a list of operand ranges, what is the subportion of the iteration
        space involved in the computation".
        This is the inverse problem of `getLoopsToShapesMap`.
        Return the empty AffineMap when such an AffineMap cannot be constructed.
        The default behavior is based on a very simple inference procedure that
        only works with permutation affine maps.
        A more advanced Tensor-Comprehension like inference is possible but has
        proven to be ambiguous in unfavorable case.
        A safer and more robust alternative is to allow each op to define
        its own AffineMap.
      }],
      /*retTy=*/"mlir::AffineMap",
      /*methodName=*/"getShapesToLoopsMap",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return inversePermutation(getLoopsToShapesMap());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` rank or zero for scalars or vectors not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // Tensor and Memref container types have a rank.
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getRank();
        }
        return 0;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the `opOperand` shape or an empty vector for scalars or vectors
        not wrapped within a tensor or a memref.
      }],
      /*retTy=*/"mlir::ArrayRef<int64_t>",
      /*methodName=*/"getShape",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getShape();
        }
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Like `getShape`, but only returns statically-known information, without
        generating any new IR. For each shape dimension, returns >=0 if that
        dimension is statically known, or ShapedType::kDynamic otherwise.
      }],
      /*retTy=*/"mlir::SmallVector<int64_t>",
      /*methodName=*/"getStaticShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::SmallVector<int64_t> res;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(res, getShape(&opOperand));
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if any of the operands has a dynamic shape.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasDynamicShape",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the input or output indexing map for `opOperand`.
      }],
      /*retTy=*/"mlir::AffineMap",
      /*methodName=*/"getMatchingIndexingMap",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for a `result`.
      }],
      /*retTy=*/"mlir::AffineMap",
      /*methodName=*/"getIndexingMapMatchingResult",
      /*args=*/(ins "OpResult":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          $_op.getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + $_op.getNumDpsInputs() +
                 result.getResultNumber());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return yes if an operation has an index semantics
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasIndexSemantics",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];
  let verify = [{ return detail::verifyStructuredOpInterface($_op); }];
  let extraClassDeclaration = [{
    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    mlir::SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    mlir::SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    mlir::SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    mlir::SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    llvm::LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    /// Return the index in the indexingMaps vector that corresponds to this `opOperand`
    int64_t getIndexingMapIndex(OpOperand *opOperand);
  }];
}

def HIVMUnitFlagEnabledInterface : OpInterface<"HIVMUnitFlagEnabled"> {
  let cppNamespace = "::mlir::hivm";
  let description = [{
    unit-flag enabled operators interface.
  }];
  let methods = [
    //===------------------------------------------------------------------===//
    // Operation property handling.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return unit-flag input value. It should be manually added to the 
        Operator input arguments named 'unit_flag_mode' and assembly format. 
      }],
      /*retTy=*/"::std::optional<::mlir::hivm::UnitFlagAttr>",
      /*methodName=*/"getUnitFlagModeValue",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return unit-flag input value. It should be manually added to the 
        Operator input arguments named 'unit_flag_mode' and assembly format. 
      }],
      /*retTy=*/"::std::optional<::mlir::Value>",
      /*methodName=*/"getUnitFlagModeCondition",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return unit-flag mode value needed to be passed to the standard library 
        call. It should be one of the possible values for unit-flag (0b00, 0b10, 0b11).
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getUnitFlagModeLibValue",
      /*args=*/(ins "::mlir::PatternRewriter&":$rewriter),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::hivm::detail::getUnitFlagModeLibValueImpl($_op, rewriter);
      }]
    >];
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_TD
