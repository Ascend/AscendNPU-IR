//===- RegbaseUtils.cpp - Utilities to support the HIVM dialect
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the HIVM dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::hivm;

bool hivm_regbaseintrins::isaVstsInstrOp(Operation *op) {
  return llvm::isa<hivm_regbaseintrins::Vstsx1V64S32InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V64U32InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V128S16InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V128U16InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V256S8InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V256U8InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V64F32InstrOp>(op) ||
         llvm::isa<hivm_regbaseintrins::Vstsx1V128F16InstrOp>(op);
}

Operation *hivm_regbaseintrins::findPredicateMapOp(Value result,
                                                   PatternRewriter &rewriter,
                                                   Location loc,
                                                   VectorType argVectorType) {
  while (true) {
    if (result.use_empty()) {
      return nullptr;
    }
    Operation *user = *(result.getUsers().begin());
    if (!user || llvm::isa<func::ReturnOp, scf::YieldOp>(user)) {
      return nullptr;
    }
    if (llvm::isa<mlir::vector::TransferWriteOp>(user)) {
      // Find the operand of the TransferWrite op that is a mask op
      for (Value operand : user->getOperands()) {
        Operation *defineOp = operand.getDefiningOp();
        if (isa_and_nonnull<mlir::vector::CreateMaskOp>(defineOp) ||
            isa_and_nonnull<mlir::vector::ConstantMaskOp>(defineOp)) {
          // After finding the mask op, check if the mask already has a cast
          assert(defineOp != nullptr);
          assert(result.getDefiningOp() != nullptr);
          if (defineOp->getBlock() != result.getDefiningOp()->getBlock() ||
              !defineOp->isBeforeInBlock(result.getDefiningOp())) {
            return nullptr;
          }
          for (Operation *maskUser : operand.getUsers()) {
            if (isa<UnrealizedConversionCastOp>(maskUser)) {
              // If the mask has already been casted, use that cast
              return maskUser;
            }
          }
          // Otherwise, create a new UnrealizedConversionCast op
          auto resultType = LLVM::getFixedVectorType(rewriter.getI1Type(), 256);
          return rewriter.create<UnrealizedConversionCastOp>(loc, resultType,
                                                             operand);
        }
        mlir::vector::TransferWriteOp transfer =
            cast<mlir::vector::TransferWriteOp>(user);
        if (isa_and_nonnull<UnrealizedConversionCastOp>(defineOp) &&
            transfer.getMask() &&
            transfer.getMask().getDefiningOp() == defineOp) {
          if (transfer.getVector().getType() == argVectorType) {
            auto resultType =
                LLVM::getFixedVectorType(rewriter.getI1Type(), 256);
            return rewriter.create<UnrealizedConversionCastOp>(loc, resultType,
                                                               operand);
          }
        }
      }
      return nullptr;
    }
    if (isaVstsInstrOp(user)) {
      // Find the operand of the vector store op that is a
      // UnrealizedConversionCast op
      for (auto operand : user->getOperands()) {
        Operation *defineOp = operand.getDefiningOp();
        if (isa_and_nonnull<UnrealizedConversionCastOp>(defineOp)) {
          // After finding the cast op, find the original mask
          while (llvm::isa<UnrealizedConversionCastOp>(defineOp)) {
            assert(defineOp != nullptr);
            Value o = defineOp->getOperands().front();
            defineOp = o.getDefiningOp();
          }
          // Once defineOp is not a cast op, it must be the original mask
          return defineOp;
        }
      }
    }

    // If a vector store op or TransferWrite op is not found, continue
    // traversing the user chain
    result = user->getResult(0);
  }
}

// Gets a suitable vector size from the element type, assuming the type is
// supported by davinci metal. Vector size is fixed for Regbase, returns
// an integer representing a vector size,
int64_t hivm_regbaseintrins::getVectorSizeByElementType(Type t) {
  int factor = t.isInteger(64) ? 2 : 1;
  constexpr unsigned int vectorByteLength = 256;
  constexpr unsigned int byteSize = 8;
  return factor * vectorByteLength /
         static_cast<int>((t.getIntOrFloatBitWidth() / byteSize));
}

Operation *hivm_regbaseintrins::buildVBrOp(Value broadcastSource,
                                           Type elementType,
                                           PatternRewriter &rewriter) {
  int64_t vecSize = getVectorSizeByElementType(elementType);
  VectorType resVecType = VectorType::get({vecSize}, elementType);

  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
      elementType.isUnsignedInteger(32) || elementType.isSignedInteger(16) ||
      elementType.isSignlessInteger(16) || elementType.isUnsignedInteger(16) ||
      elementType.isSignedInteger(8) || elementType.isSignlessInteger(8) ||
      elementType.isUnsignedInteger(8) || elementType.isF32() ||
      elementType.isF16() || elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::VbrInstrOp>(
        rewriter.getUnknownLoc(), resVecType, broadcastSource);
  llvm_unreachable("Invalid vbr element type.");
}

Operation *hivm_regbaseintrins::buildVldsOp(Value basePtr, Value offset,
                                            Type elementType,
                                            PatternRewriter &rewriter,
                                            unsigned int dist,
                                            bool postUpdate) {

  assert(dist <= 17 && "#dist is defined from 0 to 17 by hardware");

  Value distVal = rewriter.create<arith::ConstantOp>(
      basePtr.getLoc(), rewriter.getI32IntegerAttr(dist));

  Value pVal = rewriter.create<arith::ConstantOp>(
      basePtr.getLoc(),
      rewriter.getI32IntegerAttr(static_cast<int>(postUpdate)));

  int64_t vecSize = getVectorSizeByElementType(elementType);
  VectorType resVecType = VectorType::get({vecSize}, elementType);

  // putting constant zero for both #dist and #postUpdateEnable
  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V64S32InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isUnsignedInteger(32))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V64U32InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isSignedInteger(16) || elementType.isSignlessInteger(16))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V128S16InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isUnsignedInteger(16))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V128U16InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V256S8InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isUnsignedInteger(8))
    return rewriter.create<hivm_regbaseintrins::Vldsx1V256U8InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isF32())
    return rewriter.create<hivm_regbaseintrins::Vldsx1V64F32InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isF16())
    return rewriter.create<hivm_regbaseintrins::Vldsx1V128F16InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::Vldsx1V128BF16InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::Vldsx1V256F8E4InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  if (elementType.isFloat8E5M2())
    return rewriter.create<hivm_regbaseintrins::Vldsx1V256F8E5InstrOp>(
        basePtr.getLoc(), resVecType, basePtr, offset, distVal, pVal);
  llvm_unreachable("Invalid vlds element type.");
}

Operation *hivm_regbaseintrins::buildVldusOp(Value basePtr, Value alignData,
                                             Type elementType,
                                             PatternRewriter &rewriter) {
  int64_t vecSize = getVectorSizeByElementType(elementType);
  VectorType resVecType = VectorType::get({vecSize}, elementType);
  VectorType resType = VectorType::get({32}, rewriter.getI8Type());
  Type resultTypes = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {resVecType, resType, basePtr.getType()});
  auto loc = basePtr.getLoc();

  if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8))
    return rewriter.create<hivm_regbaseintrins::VldusS8InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isUnsignedInteger(8))
    return rewriter.create<hivm_regbaseintrins::VldusU8InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isSignedInteger(16) || elementType.isSignlessInteger(16))
    return rewriter.create<hivm_regbaseintrins::VldusS16InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isUnsignedInteger(16))
    return rewriter.create<hivm_regbaseintrins::VldusU16InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32))
    return rewriter.create<hivm_regbaseintrins::VldusS32InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isUnsignedInteger(32))
    return rewriter.create<hivm_regbaseintrins::VldusU32InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isSignedInteger(64) || elementType.isSignlessInteger(64))
    return rewriter.create<hivm_regbaseintrins::VldusS64InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::VldusF8E4InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isFloat8E5M2())
    return rewriter.create<hivm_regbaseintrins::VldusF8E5InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::VldusBF16InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isF16())
    return rewriter.create<hivm_regbaseintrins::VldusF16InstrOp>(
        loc, resultTypes, basePtr, alignData);
  if (elementType.isF32())
    return rewriter.create<hivm_regbaseintrins::VldusF32InstrOp>(
        loc, resultTypes, basePtr, alignData);
  std::string res = "";
  llvm::raw_string_ostream msg(res);
  elementType.print(msg);
  llvm_unreachable(("Invalid vldus element type: " + msg.str()).c_str());
}

Operation *hivm_regbaseintrins::buildVldusPostOp(Value basePtr, Value alignData,
                                                 Value cceVL, Type elementType,
                                                 PatternRewriter &rewriter) {
  int64_t vecSize = getVectorSizeByElementType(elementType);
  VectorType resVecType = VectorType::get({vecSize}, elementType);
  VectorType resType = VectorType::get({32}, rewriter.getI8Type());
  Type resultTypes = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {resVecType, resType, basePtr.getType()});
  auto loc = basePtr.getLoc();

  if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8))
    return rewriter.create<hivm_regbaseintrins::VldusPostS8InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isUnsignedInteger(8))
    return rewriter.create<hivm_regbaseintrins::VldusPostU8InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isSignedInteger(16) || elementType.isSignlessInteger(16))
    return rewriter.create<hivm_regbaseintrins::VldusPostS16InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isUnsignedInteger(16))
    return rewriter.create<hivm_regbaseintrins::VldusPostU16InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32))
    return rewriter.create<hivm_regbaseintrins::VldusPostS32InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isUnsignedInteger(32))
    return rewriter.create<hivm_regbaseintrins::VldusPostU32InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isSignedInteger(64) || elementType.isSignlessInteger(64))
    return rewriter.create<hivm_regbaseintrins::VldusPostS64InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::VldusPostF8E4InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isFloat8E5M2())
    return rewriter.create<hivm_regbaseintrins::VldusPostF8E5InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::VldusPostBF16InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isF16())
    return rewriter.create<hivm_regbaseintrins::VldusPostF16InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  if (elementType.isF32())
    return rewriter.create<hivm_regbaseintrins::VldusPostF32InstrOp>(
        loc, resultTypes, basePtr, alignData, cceVL);
  std::string res = "";
  llvm::raw_string_ostream msg(res);
  elementType.print(msg);
  llvm_unreachable(("Invalid vldus post element type: " + msg.str()).c_str());
}

Operation *hivm_regbaseintrins::buildVldasOp(Value basePtr,
                                             PatternRewriter &rewriter) {
  VectorType resVecType = VectorType::get({32}, rewriter.getI8Type());
  auto loc = basePtr.getLoc();
  return rewriter.create<hivm_regbaseintrins::VldasInstrOp>(loc, resVecType,
                                                            basePtr);
}

Operation *hivm_regbaseintrins::buildVstsOp(Value srcRegister, Value basePtr,
                                            Value offset, Value predicateVector,
                                            PatternRewriter &rewriter,
                                            unsigned int dist) {

  Value cstZero = rewriter.create<arith::ConstantOp>(
      basePtr.getLoc(), rewriter.getI32IntegerAttr(0));
  VectorType vecType = cast<VectorType>(srcRegister.getType());
  Type elementType = vecType.getElementType();
  ArrayRef vectorDimensions = vecType.getShape();
  unsigned numElements = 1;
  // Use the vector shape to determine the number of elements
  for (auto dim : vectorDimensions) {
    // numElements should not change until the final dimension
    assert(numElements == 1 && "Unsupported vector shape.");
    if (dim != 1) {
      numElements = static_cast<unsigned>(dim);
    }
  }
  // VstsOp only supports 1D vectors
  if (vecType.getShape().size() > 1) {
    // If the vector is multidimensional, collapse into a 1D vector using an
    // unrealized conversion cast
    VectorType resVecType = VectorType::get({numElements}, elementType);
    Value newSrcRegister = rewriter
                               .create<UnrealizedConversionCastOp>(
                                   basePtr.getLoc(), resVecType, srcRegister)
                               .getResult(0);
    srcRegister = newSrcRegister;
  }
  if (numElements == 1) {
    int64_t vecSize = getVectorSizeByElementType(elementType);
    VectorType resVecType = VectorType::get({vecSize}, elementType);
    Value newSrcRegister = rewriter
                               .create<UnrealizedConversionCastOp>(
                                   basePtr.getLoc(), resVecType, srcRegister)
                               .getResult(0);
    srcRegister = newSrcRegister;
  }
  Value distVal;
  if (dist) {
    distVal = rewriter.create<arith::ConstantOp>(
        basePtr.getLoc(), rewriter.getI32IntegerAttr(dist));
  } else {
    auto elementTypeNumBit = elementType.getIntOrFloatBitWidth();
    switch (elementTypeNumBit) {
    case 8:
      distVal = rewriter.create<arith::ConstantOp>(
          basePtr.getLoc(),
          rewriter.getI32IntegerAttr(numElements == 1 ? 3 : 0));
      break;
    case 16:
      distVal = rewriter.create<arith::ConstantOp>(
          basePtr.getLoc(),
          rewriter.getI32IntegerAttr(numElements == 1 ? 4 : 1));
      break;
    case 32:
      distVal = rewriter.create<arith::ConstantOp>(
          basePtr.getLoc(),
          rewriter.getI32IntegerAttr(numElements == 1 ? 5 : 2));
      break;
    default:
      llvm_unreachable("unsupported datatypes");
    }
  }

  // Putting constant zero for both #dist and #postUpdateEnable
  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V64S32InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isUnsignedInteger(32))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V64U32InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isSignedInteger(16) || elementType.isSignlessInteger(16))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V128S16InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isUnsignedInteger(16))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V128U16InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V256S8InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isUnsignedInteger(8))
    return rewriter.create<hivm_regbaseintrins::Vstsx1V256U8InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isF32())
    return rewriter.create<hivm_regbaseintrins::Vstsx1V64F32InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isF16())
    return rewriter.create<hivm_regbaseintrins::Vstsx1V128F16InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::Vstsx1V128BF16InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::Vstsx1V256F8E4InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  if (elementType.isFloat8E5M2())
    return rewriter.create<hivm_regbaseintrins::Vstsx1V256F8E5InstrOp>(
        basePtr.getLoc(), srcRegister, basePtr, offset, distVal, cstZero,
        predicateVector);
  llvm_unreachable("Invalid vsts element type.");
}

Operation *hivm_regbaseintrins::buildVstusPostOp(Value srcRegister,
                                                 Value basePtr, Value eleSize,
                                                 Value tmp,
                                                 PatternRewriter &rewriter) {
  VectorType vecType = cast<VectorType>(srcRegister.getType());
  Type elementType = vecType.getElementType();
  ArrayRef vectorDimensions = vecType.getShape();
  auto loc = basePtr.getLoc();
  unsigned numElements = 1;
  // Use the vector shape to determine the number of elements
  for (auto dim : vectorDimensions) {
    // numElements should not change until the final dimension
    assert(numElements == 1 && "Unsupported vector shape.");
    if (dim != 1) {
      numElements = static_cast<unsigned>(dim);
    }
  }
  // VstusOp only supports 1D vectors
  if (vecType.getShape().size() > 1) {
    // If the vector is multidimensional, collapse into a 1D vector using an
    // unrealized conversion cast
    VectorType resVecType = VectorType::get({numElements}, elementType);
    Value newSrcRegister =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, resVecType, srcRegister)
            .getResult(0);
    srcRegister = newSrcRegister;
  }
  if (numElements == 1) {
    int64_t vecSize = getVectorSizeByElementType(elementType);
    VectorType resVecType = VectorType::get({vecSize}, elementType);
    Value newSrcRegister =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, resVecType, srcRegister)
            .getResult(0);
    srcRegister = newSrcRegister;
  }

  VectorType resType = VectorType::get({32}, rewriter.getI8Type());
  Type resultTypes = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {resType, basePtr.getType()});

  if (elementType.isSignedInteger(8) || elementType.isSignlessInteger(8))
    return rewriter.create<hivm_regbaseintrins::VstusPostS8InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isUnsignedInteger(8))
    return rewriter.create<hivm_regbaseintrins::VstusPostU8InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isSignedInteger(16) || elementType.isSignlessInteger(16))
    return rewriter.create<hivm_regbaseintrins::VstusPostS16InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isUnsignedInteger(16))
    return rewriter.create<hivm_regbaseintrins::VstusPostU16InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32))
    return rewriter.create<hivm_regbaseintrins::VstusPostS32InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isUnsignedInteger(32))
    return rewriter.create<hivm_regbaseintrins::VstusPostU32InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isSignedInteger(64) || elementType.isSignlessInteger(64))
    return rewriter.create<hivm_regbaseintrins::VstusPostS64InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isF32())
    return rewriter.create<hivm_regbaseintrins::VstusPostF32InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isF16())
    return rewriter.create<hivm_regbaseintrins::VstusPostF16InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::VstusPostBF16InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::VstusPostF8E4InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  if (elementType.isFloat8E5M2())
    return rewriter.create<hivm_regbaseintrins::VstusPostF8E5InstrOp>(
        loc, resultTypes, srcRegister, basePtr, eleSize, tmp);
  std::string res = "";
  llvm::raw_string_ostream msg(res);
  elementType.print(msg);
  llvm_unreachable(("Invalid vstus element type: " + msg.str()).c_str());
}

Operation *hivm_regbaseintrins::buildVstasOp(Value srcRegister, Value basePtr,
                                             PatternRewriter &rewriter) {
  auto loc = basePtr.getLoc();
  Value cstZero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  return rewriter.create<hivm_regbaseintrins::VstasInstrOp>(
      loc, srcRegister, basePtr, cstZero, cstZero);
}

Operation *hivm_regbaseintrins::buildPsetOp(Value pattern,
                                            unsigned elementBitWidth,
                                            PatternRewriter &rewriter) {
  VectorType resVecType = VectorType::get({256}, rewriter.getI1Type());
  unsigned elementByteCount = (elementBitWidth) / 8;

  // set #postUpdateEnable to zero
  switch (elementByteCount) {
  case 1:
    return rewriter.create<hivm_regbaseintrins::PgeB8>(
        pattern.getLoc(), resVecType, pattern, pattern);
  case 2:
    return rewriter.create<hivm_regbaseintrins::PgeB16>(
        pattern.getLoc(), resVecType, pattern, pattern);
  case 3:
  case 4:
    return rewriter.create<hivm_regbaseintrins::PgeB32>(
        pattern.getLoc(), resVecType, pattern, pattern);
  }
  llvm_unreachable("Invalid element bit width for a predicate vector.");
}

Operation *hivm_regbaseintrins::buildMovvpOp(Location loc, Type type,
                                             Value extractOp,
                                             PatternRewriter &rewriter,
                                             int elementAlignment) {
  auto pregType = VectorType::get(256, rewriter.getI1Type());
  Value cstZero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
  Operation *result;
  switch (elementAlignment) {
  case 16:
  case 32:
    result = rewriter.create<MovvpInstrOp>(loc, pregType, extractOp, cstZero);
    break;
  default:
    llvm_unreachable("elementAlignment is not supported");
  }
  return result;
}

Operation *hivm_regbaseintrins::buildPpackOp(Location loc, Value part,
                                             Value src,
                                             PatternRewriter &rewriter) {
  auto pregType = VectorType::get(256, rewriter.getI1Type());
  Operation *result = rewriter.create<PpackInstrOp>(loc, pregType, src, part);
  return result;
}

Operation *hivm_regbaseintrins::buildPunpackOp(Location loc, Value part,
                                               Value src,
                                               PatternRewriter &rewriter) {
  auto pregType = VectorType::get(256, rewriter.getI1Type());
  Operation *result = rewriter.create<PunpackInstrOp>(loc, pregType, src, part);
  return result;
}

Operation *hivm_regbaseintrins::buildVpackOp(Location loc, Value part,
                                             Value src, Type vectorType,
                                             PatternRewriter &rewriter) {
  // Check if src has a valid vector type with b16 or b32 element type
  if (auto vecType = dyn_cast<VectorType>(src.getType())) {
    Type elemType = vecType.getElementType();
    unsigned bitWidth = elemType.getIntOrFloatBitWidth();
    if (bitWidth != 16 && bitWidth != 32) {
      llvm_unreachable("Invalid element type for VpackInstrOp, only b16 and b32 are supported");
    }
  }
  Operation *result = rewriter.create<VpackInstrOp>(loc, vectorType, src, part);
  return result;
}

Operation *hivm_regbaseintrins::buildVunpackOp(Location loc, Value part,
                                                Value src, Type vectorType,
                                                PatternRewriter &rewriter) {
  // Check if src has a valid vector type with b8, b16, or b6 element type
  if (auto vecType = dyn_cast<VectorType>(src.getType())) {
    Type elemType = vecType.getElementType();
    unsigned bitWidth = elemType.getIntOrFloatBitWidth();
    if (bitWidth != 8 && bitWidth != 16 && bitWidth != 6) {
      llvm_unreachable("Invalid element type for Vzunpack, only b8, b16, and b6 are supported");
    }
  }
  Operation *result = rewriter.create<Vzunpack>(loc, vectorType, src, part);
  return result;
}

Operation *hivm_regbaseintrins::buildPltOp(Value count,
                                           unsigned elementBitWidth,
                                           PatternRewriter &rewriter) {
  VectorType resVecType = VectorType::get({256}, rewriter.getI1Type());
  Value cstZero = rewriter.create<arith::ConstantOp>(
      count.getLoc(), rewriter.getI16IntegerAttr(0));

  unsigned elementByteCount = (elementBitWidth + 7) / 8;
  Location loc = count.getLoc();
  switch (elementByteCount) {
  case 1:
    return rewriter.create<hivm_regbaseintrins::PltMB8>(loc, resVecType,
                                                        cstZero, count);
  case 2:
    return rewriter.create<hivm_regbaseintrins::PltMB16>(loc, resVecType,
                                                         cstZero, count);
  case 3:
  case 4:
    return rewriter.create<hivm_regbaseintrins::PltMB32>(loc, resVecType,
                                                         cstZero, count);
  }
  llvm_unreachable("Invalid element bit width for a predicate vector.");
}

Operation *hivm_regbaseintrins::buildPstuOp(Value data, Value dataPtr,
                                            PatternRewriter &rewriter,
                                            int elementAlignment,
                                            Value alignData) {
  auto loc = dataPtr.getLoc();
  VectorType resType = VectorType::get({32}, rewriter.getI8Type());
  Type resultTypes = LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {resType, dataPtr.getType()});
  Operation *result;
  switch (elementAlignment) {
  case 16:
    result = rewriter.create<PSTUStoreS16InstOp>(loc, resultTypes, data,
                                                 dataPtr, alignData);
    break;
  case 32:
    result = rewriter.create<PSTUStoreS32InstOp>(loc, resultTypes, data,
                                                 dataPtr, alignData);
    break;
  default:
    llvm_unreachable("Invalid elementAlignment!");
  }
  return result;
}

Operation *hivm_regbaseintrins::buildVdupOp(Value srcRegister,
                                            Value predicateVector,
                                            Type elementType,
                                            PatternRewriter &rewriter) {
  Value cstOne = rewriter.create<arith::ConstantOp>(
      srcRegister.getLoc(), rewriter.getI32IntegerAttr(1));
  int64_t vecSize = getVectorSizeByElementType(elementType);
  VectorType resVecType = VectorType::get({vecSize}, elementType);

  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
      elementType.isUnsignedInteger(32) || elementType.isSignedInteger(16) ||
      elementType.isSignlessInteger(16) || elementType.isUnsignedInteger(16) ||
      elementType.isSignedInteger(8) || elementType.isSignlessInteger(8) ||
      elementType.isUnsignedInteger(8) || elementType.isF32() ||
      elementType.isF16() || elementType.isBF16() ||
      elementType.isFloat8E5M2() || elementType.isFloat8E4M3FN())
    return rewriter.create<hivm_regbaseintrins::VdupZInstrOp>(
        srcRegister.getLoc(), resVecType, srcRegister, predicateVector, cstOne);
  llvm_unreachable("Invalid vsts element type.");
}

// Checks if the type is supported by Regbase
bool hivm_regbaseintrins::isHIVMSupportedType(Type t) {
  if (isa<IntegerType>(t)) {
    unsigned int width = cast<IntegerType>(t).getWidth();

    // supports 8/16/32 bits signed/unsigned integers
    return width == 8 || width == 16 || width == 32;
  }
  return isa<Float16Type>(t) || isa<Float32Type>(t);
}

namespace {
// adapted from ../ArithToHIVMLLVM/ArithToHIVMLLVM.cpp
template <typename Intr_s, typename Intr_u>
Operation *buildRegbaseBinaryVectorOp(Value lhs, Value rhs, Value pred,
                                      PatternRewriter &rewriter) {
  Type elementType = cast<VectorType>(lhs.getType()).getElementType();
  int64_t vecSize =
      mlir::hivm_regbaseintrins::getVectorSizeByElementType(elementType);
  VectorType resType = VectorType::get({vecSize}, elementType);

  if constexpr (!std::is_same_v<Intr_s, void>) {
    if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
        elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
        elementType.isSignedInteger(8) || elementType.isSignlessInteger(8) ||
        elementType.isF32() || elementType.isF16())
      return rewriter.create<Intr_s>(lhs.getLoc(), resType, lhs, rhs, pred);
  }
  if constexpr (!std::is_same_v<Intr_u, void>) {
    if (elementType.isUnsignedInteger(32) ||
        elementType.isUnsignedInteger(16) || elementType.isUnsignedInteger(8))
      return rewriter.create<Intr_u>(lhs.getLoc(), resType, lhs, rhs, pred);
  }
  llvm_unreachable("Invalid david op element type.");
}

template <typename Intr_s, typename Intr_u>
Operation *buildRegbaseUnaryVectorOp(Value lhs, Value pred,
                                     PatternRewriter &rewriter) {
  Type elementType = cast<VectorType>(lhs.getType()).getElementType();
  int64_t vecSize =
      mlir::hivm_regbaseintrins::getVectorSizeByElementType(elementType);
  VectorType resType = VectorType::get({vecSize}, elementType);

  if constexpr (!std::is_same_v<Intr_s, void>) {
    if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
        elementType.isSignedInteger(16) || elementType.isSignlessInteger(16) ||
        elementType.isF32() || elementType.isF16()) {
      return rewriter.create<Intr_s>(lhs.getLoc(), resType, lhs, pred);
    }
  }
  if constexpr (!std::is_same_v<Intr_u, void>) {
    if (elementType.isUnsignedInteger(32) ||
        elementType.isUnsignedInteger(16)) {
      return rewriter.create<Intr_u>(lhs.getLoc(), resType, lhs, pred);
    }
  }
  llvm_unreachable("Invalid david op element type.");
}
} // end anonymous namespace

Operation *hivm_regbaseintrins::buildVselOp(Value srcReg1, Value srcReg2,
                                            Value predicateVector,
                                            PatternRewriter &rewriter) {
  Type elementType = cast<VectorType>(srcReg1.getType()).getElementType();
  VectorType predicateVecType = cast<VectorType>(predicateVector.getType());
  ArrayRef predicateVectorDimensions = predicateVecType.getShape();
  unsigned numElements = 1;
  // Use the vector shape to determine the number of elements
  for (auto dim : predicateVectorDimensions) {
    // numElements should not change until the final dimension
    assert(numElements == 1 && "Unsupported vector shape.");
    if (dim != 1) {
      numElements = static_cast<unsigned>(dim);
    }
  }
  if (numElements != 256) {
    // If the mask has the wrong shape, cast it to a vector<256xi1>
    VectorType resVecType = VectorType::get({256}, rewriter.getI1Type());
    Operation *castedPredicateVector =
        rewriter.create<UnrealizedConversionCastOp>(
            rewriter.getUnknownLoc(), resVecType, predicateVector);
    predicateVector = castedPredicateVector->getResult(0);
  }

  if (elementType.isSignedInteger(32) || elementType.isSignlessInteger(32) ||
      elementType.isUnsignedInteger(32) || elementType.isSignedInteger(16) ||
      elementType.isSignlessInteger(16) || elementType.isUnsignedInteger(16) ||
      elementType.isSignedInteger(8) || elementType.isSignlessInteger(8) ||
      elementType.isUnsignedInteger(8) || elementType.isF32() ||
      elementType.isF16() || elementType.isBF16())
    return rewriter.create<hivm_regbaseintrins::VselInstrOp>(
        rewriter.getUnknownLoc(), srcReg1.getType(), srcReg1, srcReg2,
        predicateVector);
  llvm_unreachable("Invalid vsel element type.");
}

Value hivm_regbaseintrins::remove1DVectorHighDims(Location loc, Value value,
                                                  PatternRewriter &rewriter) {
  // convert vector<1x...x1x64xf32> to vector<64xf32>
  VectorType vecType = cast<VectorType>(value.getType());
  auto shape = vecType.getShape();
  if (shape.size() == 1) {
    // already 1D
    return value;
  }
  for (unsigned i = 1, n = shape.size(); i < n; ++i) {
    if (shape[i - 1] != 1) {
      llvm_unreachable("Expecting 1D vector when reducing high dimensions");
    }
  }
  auto endType =
      LLVM::getFixedVectorType(vecType.getElementType(), shape.back());
  return rewriter.create<UnrealizedConversionCastOp>(loc, endType, value)
      .getResult(0);
}
Operation *hivm_regbaseintrins::buildAddOp(Value lhs, Value rhs, Value pred,
                                           PatternRewriter &rewriter) {
  return buildRegbaseBinaryVectorOp<
      /*Intr_s*/ hivm_regbaseintrins::VaddSXInstrOp,
      /*Intr_u*/ hivm_regbaseintrins::VaddUXInstrOp>(lhs, rhs, pred, rewriter);
}
Operation *hivm_regbaseintrins::buildCumulateAddOp(Value lhs, Value pred,
                                                   PatternRewriter &rewriter) {
  return buildRegbaseUnaryVectorOp<
      /*Intr_s*/ hivm_regbaseintrins::VcaddSXInstrOp,
      /*Intr_u*/ hivm_regbaseintrins::VcaddUXInstrOp>(lhs, pred, rewriter);
}

Operation *hivm_regbaseintrins::buildMaxOp(Value lhs, Value rhs, Value pred,
                                           PatternRewriter &rewriter) {
  return buildRegbaseBinaryVectorOp<
      /*Intr_s*/ hivm_regbaseintrins::VmaxSXInstrOp,
      /*Intr_u*/ hivm_regbaseintrins::VmaxUXInstrOp>(lhs, rhs, pred, rewriter);
}
Operation *hivm_regbaseintrins::buildCumulateMaxOp(Value lhs, Value pred,
                                                   PatternRewriter &rewriter) {
  return buildRegbaseUnaryVectorOp<
      /*Intr_s*/ hivm_regbaseintrins::VcmaxSXInstrOp,
      /*Intr_u*/ hivm_regbaseintrins::VcmaxUXInstrOp>(lhs, pred, rewriter);
}

int hivm_regbaseintrins::getVldsBrcDist(const int &dataBitWidth) {
  int dist = -1;
  switch (dataBitWidth) {
  case 8:
    dist = 1;
    break;
  case 16:
    dist = 2;
    break;
  case 32:
    dist = 3;
    break;
  default:
    llvm_unreachable("Invalid data bit length!");
  }
  return dist;
}

VectorType
hivm_regbaseintrins::adjustTypeToDavidSupportedType(VectorType vecTy) {
  Type elemTy = vecTy.getElementType();
  unsigned vecSize = static_cast<unsigned>(getVectorSizeByElementType(elemTy));
  return VectorType::get(vecSize, elemTy);
}

Value hivm_regbaseintrins::createPredicateVector(Operation *op,
                                                 VectorType vecType,
                                                 PatternRewriter &rewriter) {
  Operation *boundOp = nullptr;
  Operation *afterBoundOp = nullptr;
  Operation *pltOp = nullptr;
  Operation *firstOp = nullptr;
  Operation *funcOp = op;
  Type elementType = vecType.getElementType();
  unsigned bitwidth = 0;
  // Use the element type to determine the bitwidth
  bitwidth = elementType.getIntOrFloatBitWidth();
  ArrayRef vectorDimensions = vecType.getShape();
  unsigned numElements = 1;
  // Use the vector shape to determine the number of elements
  for (auto dim : vectorDimensions) {
    // numElements should not change until the final dimension
    assert(numElements == 1 && "Unsupported vector shape.");
    if (dim != 1) {
      numElements = static_cast<unsigned>(dim);
    }
  }
  // Find the outermost operation (the funcOp)
  while (funcOp && !llvm::isa<func::FuncOp>(funcOp)) {
    funcOp = funcOp->getParentOp();
  }
  // Traverse funcOp to search for ops that can be reused
  assert(funcOp != nullptr);
  funcOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumRegions() > 0 && firstOp) {
      // Do not try to reuse ops from within blocks/regions
      return WalkResult::skip();
    }
    if (!firstOp && !isa<func::FuncOp>(op)) {
      firstOp = op;
    }
    if (boundOp && !afterBoundOp) {
      afterBoundOp = op;
    }
    if (arith::ConstantIndexOp cint = dyn_cast<arith::ConstantIndexOp>(op)) {
      if (!boundOp && cint.value() == numElements) {
        boundOp = op;
      }
    }
    if (((isa<hivm_regbaseintrins::PltB8>(op) && bitwidth == 8) ||
         (isa<hivm_regbaseintrins::PltB16>(op) && bitwidth == 16) ||
         (isa<hivm_regbaseintrins::PltB32>(op) && bitwidth == 32)) &&
        numElements != 1) {
      pltOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  // If a suitable pltOp was found, reuse it
  if (pltOp) {
    return pltOp->getResult(0);
  }
  // If a boundOp was not found, create one
  if (!boundOp) {
    rewriter.setInsertionPoint(firstOp);
    boundOp = rewriter.create<arith::ConstantIntOp>(
        op->getLoc(), rewriter.getI32Type(), numElements);
  } else {
    // Set the rewriter to point directly after the boundOp
    rewriter.setInsertionPoint(afterBoundOp);
  }
  Type srcBoundType = boundOp->getResultTypes().front();
  // Adjust the boundOp if necessary
  if (!srcBoundType.isInteger(32)) {
    Operation *convertedBoundValue = rewriter.create<arith::IndexCastOp>(
        boundOp->getLoc(), rewriter.getI32Type(), boundOp->getResult(0));
    boundOp = convertedBoundValue;
  }
  // Create the pltOp
  pltOp = buildPltOp(boundOp->getResult(0), bitwidth, rewriter);
  // Set the rewriter to point at the original op
  rewriter.setInsertionPoint(op);

  return pltOp->getResult(0);
}

VectorType hivm_regbaseintrins::createVLVectorType(Type elementType) {
  int vectorLength;
  switch (elementType.getIntOrFloatBitWidth()) {
  case 1:
  case 8:
    vectorLength = 256;
    break;
  case 16:
    vectorLength = 128;
    break;
  case 32:
    vectorLength = 64;
    break;
  case 64:
    vectorLength = 32;
    break;
  default:
    llvm_unreachable("unsupported datatype");
  }
  return VectorType::get({vectorLength}, elementType);
}

bool mlir::hivm_regbaseintrins::deviceFuncHasVFCalls(func::CallOp caller,
                                                     SymbolTable &symtab) {
  // Detect the following situation:
  //
  // device_entry_kernel(gm *ptr) {
  //    hir.matmul
  //    call xyz(ptr);
  // }
  // xyz(gm *ptr) {
  //    call outlined_vf
  // }
  //
  // That is, the body of xyz only contains calls to VF. In this case, DMA,
  // set/waits will happen within xyz as the gm ptr is passed into xyz.
  // Nothing needs to be done for xyz at the device_entry_kernel level in
  // terms of synchronization.
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(caller.getCallableForCallee());
  if (!sym)
    return false;
  func::FuncOp funcOp = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(caller, sym));
  auto result = funcOp.walk([&](func::CallOp callOp) {
    auto callee = symtab.lookup<func::FuncOp>(callOp.getCallee());
    if (!callee->hasAttr(hivm::VectorFunctionAttr::name)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt())
    return false;
  return true;
}

Value hivm_regbaseintrins::getVLRegValueOrSelf(Value orig, OpBuilder &b) {
  VectorType vType = cast<VectorType>(orig.getType());
  Type elementType = vType.getElementType();
  VectorType vlType;
  if (elementType.isInteger(1)) {
    vlType = VectorType::get(SmallVector<int64_t>{util::PREDICATE_BITS},
                             elementType);
  } else {
    vlType = hivm_regbaseintrins::createVLVectorType(elementType);
  }

  if (vType == vlType)
    return orig;

  return b.create<UnrealizedConversionCastOp>(orig.getLoc(), vlType, orig)
      ->getResult(0);
}
