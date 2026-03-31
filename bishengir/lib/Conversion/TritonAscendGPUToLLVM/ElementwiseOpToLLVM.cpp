//===--TritonAscendElementwiseOpToLLVMPass.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Analysis/AscendUtility.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/PatternTritonAscendGPUOpToLLVM.h"
#include "bishengir/Conversion/TritonAscendGPUToLLVM/TargetInfo.h"
#include "bishengir/Dialect/AscendDPX/IR/AscendDPX.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::ascend;
using namespace mlir::triton::gpu;

namespace {

Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType();
  return type;
}

int getNumElementsPerThreads(Type type,
                             const LLVMTypeConverter *typeConverter) {
  int numElemsPerThread = 1;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto structType =
        dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
    if (structType)
      numElemsPerThread = static_cast<int>(structType.getBody().size());
  }
  return numElemsPerThread;
}

struct AscendSIToFPOpConversion
    : public ElementwiseOpConversionBase<arith::SIToFPOp, AscendSIToFPOpConversion> {
  using Base =
      ElementwiseOpConversionBase<arith::SIToFPOp, AscendSIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit AscendSIToFPOpConversion(
      LLVMTypeConverter &typeConverter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit) {}

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    SmallVector<Value> outVals;
    for (size_t i = 0; i < operands.size(); i++) {
      outVals.push_back(
         rewriter.create<LLVM::SIToFPOp>(loc, outElemTy, operands[i][0]));
    }
    return outVals;
  };
};

struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
  using ConvertOpToLLVMPattern<AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    if (auto resultTensorTy = dyn_cast<RankedTensorType>(resultTy)) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type resultTensorElemTy = resultTensorTy.getElementType();
      Type elemTy = typeConverter->convertType(
          cast<PointerType>(resultTensorElemTy).getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorElemTy);
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (auto [idx, ptr, offset] : llvm::enumerate(ptrs, offsets)) {
        resultVals[idx] = b.gep(ptrTy, elemTy, ptr, offset);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(isa<PointerType>(resultTy));
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          cast<PointerType>(resultTy).getPointeeType());
      Value result = b.gep(resultPtrTy, resultElemTy, adaptor.getPtr(),
                           adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp> createDestOps(arith::CmpIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) const {
    return {rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<LLVM::FCmpOp>
  createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      LLVM_DEBUG(llvm::dbgs() << "ExternElementwiseOpConversion funcName is empty");

    if (auto ascendOp =
            tryCreateAscendDPXOp(funcName, rewriter, loc, elemTy, operands))
      return {ascendOp};

    llvm_unreachable("Unknown triton::ExternElementwiseOp");
  }

private:
  Value tryCreateAscendDPXOp(StringRef funcName,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Type elemTy,
                             MultipleOperandsRange operands) const {

    if (operands[0].size() == 1) {
      Value operand = operands[0][0];

      if (auto unaryOp =
              llvm::StringSwitch<std::function<Value()>>(funcName)
                  .Case("__hmf_tanhf",
                        [&] {
                          return rewriter.create<ascend_dpx::TanhOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_tanhDh",
                        [&] {
                          return rewriter.create<ascend_dpx::TanhOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_tanf",
                        [&] {
                          return rewriter.create<ascend_dpx::TanOp>(loc, elemTy,
                                                                    operand);
                        })
                  .Case("__hmf_tanDh",
                        [&] {
                          return rewriter.create<ascend_dpx::TanOp>(loc, elemTy,
                                                                    operand);
                        })
                  .Case("__hmf_atanf",
                        [&] {
                          return rewriter.create<ascend_dpx::AtanOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_atanDh",
                        [&] {
                          return rewriter.create<ascend_dpx::AtanOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_recipf",
                        [&] {
                          return rewriter.create<ascend_dpx::RecipOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_recipDh",
                        [&] {
                          return rewriter.create<ascend_dpx::RecipOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_log1pf",
                        [&] {
                          return rewriter.create<ascend_dpx::Log1pOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_log1pDh",
                        [&] {
                          return rewriter.create<ascend_dpx::Log1pOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_ilogbf",
                        [&] {
                          return rewriter.create<ascend_dpx::ILogbOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_ilogbDh",
                        [&] {
                          return rewriter.create<ascend_dpx::ILogbOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_reluf",
                        [&] {
                          return rewriter.create<ascend_dpx::ReluOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_reluDh",
                        [&] {
                          return rewriter.create<ascend_dpx::ReluOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_roundf",
                        [&] {
                          return rewriter.create<ascend_dpx::RoundOp>(
                              loc, elemTy, operand);
                        })
                  .Case("__hmf_rint",
                        [&] {
                          return rewriter.create<ascend_dpx::RintOp>(
                              loc, elemTy, operand);
                        })
                  .Default(nullptr)) {
        return unaryOp();
      }

      if (auto checkOp =
              llvm::StringSwitch<std::function<Value()>>(funcName)
                  .Case("__hmf_isnan",
                        [&] {
                          return rewriter.create<ascend_dpx::IsNanOp>(
                              loc, rewriter.getI1Type(), operand);
                        })
                  .Case("__hmf_isinf",
                        [&] {
                          return rewriter.create<ascend_dpx::IsInfOp>(
                              loc, rewriter.getI1Type(), operand);
                        })
                  .Default(nullptr)) {
        return checkOp();
      }
    }

    if (operands[0].size() == 2) {
      Value lhs = operands[0][0];
      Value rhs = operands[0][1];

      if (auto binaryOp =
              llvm::StringSwitch<std::function<Value()>>(funcName)
                  .Case("__hmf_powf",
                        [&] {
                          return rewriter.create<ascend_dpx::PowOp>(loc, elemTy,
                                                                    lhs, rhs);
                        })
                  .Case("__hmf_powDh",
                        [&] {
                          return rewriter.create<ascend_dpx::PowOp>(loc, elemTy,
                                                                    lhs, rhs);
                        })
                  .Case("__hmf_powDb",
                        [&] {
                          return rewriter.create<ascend_dpx::PowOp>(loc, elemTy,
                                                                    lhs, rhs);
                        })
                  .Case("__hmf_powi",
                        [&] {
                          return rewriter.create<ascend_dpx::PowOp>(loc, elemTy,
                                                                    lhs, rhs);
                        })
                  .Case("__hmf_ldexpf",
                        [&] {
                          return rewriter.create<ascend_dpx::LdExpOp>(
                              loc, elemTy, lhs, rhs);
                        })
                  .Case("__hmf_ldexpDh",
                        [&] {
                          return rewriter.create<ascend_dpx::LdExpOp>(
                              loc, elemTy, lhs, rhs);
                        })
                  .Default(nullptr)) {
        return binaryOp();
      }
    }

    return nullptr;
  }
};

struct ElementwiseInlineAsmOpConversion
    : public ConvertOpToLLVMPattern<ElementwiseInlineAsmOp> {
  using Base = ConvertOpToLLVMPattern<ElementwiseInlineAsmOp>;

  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  using OpAdaptor = typename Base::OpAdaptor;

  // If operand size is smaller than 32 bits, pack in groups of 32 bits.
  SmallVector<Value> packOperands(ElementwiseInlineAsmOp op,
                                  MultipleOperandsRange operands,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> packedOperands;
    unsigned numPackedElements = op.getPackedElement();
    for (int i = 0, e = (int)op.getNumOperands(); i < e; i++) {
      Type elemTy = getElementType(op.getOperand(i));
      unsigned bitWidth =
          elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
      unsigned numElementPerReg = std::max(32 / bitWidth, 1u);
      numElementPerReg = std::min(numElementPerReg, numPackedElements);
      for (unsigned int j = 0; j < numPackedElements; j += numElementPerReg) {
        if (numElementPerReg == 1) {
          packedOperands.push_back(operands[j][i]);
          continue;
        }
        Type t =
            vec_ty(getTypeConverter()->convertType(elemTy), numElementPerReg);
        Value packed = b.undef(t);
        for (unsigned int k = 0; k < numElementPerReg; k++) {
          packed = b.insert_element(packed, operands[j + k][i], b.i32_val(k));
        }
        packedOperands.push_back(packed);
      }
    }
    return packedOperands;
  }

  SmallVector<SmallVector<Value>>
  createDestOps(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter,
                MultipleOperandsRange operands, Location loc) const {
    auto ctx = op->getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (operands.size() % op.getPackedElement() != 0)
      llvm::report_fatal_error("Inline asm op has more packed elements than "
                               "number of elements per thread.");

    // Pack elems smaller than 32 bits into 32-bit registers.
    SmallVector<Value> packedOperands =
        packOperands(op, operands, rewriter, loc);

    // Types returned by the LLVM asm op.  If there's more than one, they'll be
    // wrapped in a struct.
    SmallVector<Type> asmRetTypes;
    for (auto result : op.getResult()) {
      auto ty = getTypeConverter()->convertType(getElementType(result));

      // Pack return elements into 32-bits.
      unsigned bitWidth = ty.isIntOrFloat() ? ty.getIntOrFloatBitWidth() : 64;
      unsigned numElemsPerReg =
          std::min(std::max(32 / bitWidth, 1u), op.getPackedElement());
      assert(op.getPackedElement() % numElemsPerReg == 0);
      if (numElemsPerReg > 1) {
        ty = vec_ty(ty, numElemsPerReg);
      }
      for (unsigned i = 0; i < op.getPackedElement() / numElemsPerReg; i++) {
        asmRetTypes.push_back(ty);
      }
    }
    Type asmRetType =
        asmRetTypes.size() > 1 ? struct_ty(asmRetTypes) : asmRetTypes[0];

    Value asmResults =
        rewriter
            .create<LLVM::InlineAsmOp>(
                loc, asmRetType,
                /*operands=*/packedOperands,
                /*asm_string=*/op.getAsmString(),
                /*constraints=*/op.getConstraints(),
                /*has_side_effects=*/!op.getPure(),
                /*is_align_stack=*/false,
#if !BSPRIV_DAVINCI_BISHENGIR
                LLVM::TailCallKind::None,
#endif
                /*asm_dialect=*/
                LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                          LLVM::AsmDialect::AD_ATT),
                /*operand_attrs=*/ArrayAttr())
            ->getResult(0);

    // asmResults is a flat struct; pack its values into
    // [return_value][op.getPackedElement()].
    SmallVector<SmallVector<Value>> ret(op->getNumResults());
    int structIdx = 0;
    for (unsigned int i = 0; i < op->getNumResults(); i++) {
      for (uint32_t j = 0; j < op.getPackedElement(); j++) {
        Value val;
        if (asmRetTypes.size() > 1) {
          val = b.extract_val(asmResults, structIdx++);
        } else {
          val = asmResults;
        }
        if (auto vectorTy = dyn_cast<VectorType>(val.getType())) {
          for (int k = 0; k < vectorTy.getNumElements(); k++) {
            ret[i].push_back(b.extract_element(val, b.i32_val(k)));
          }
          j += (uint32_t)vectorTy.getNumElements() - 1;
        } else {
          ret[i].push_back(val);
        }
      }
    }
    return ret;
  }

  LogicalResult
  matchAndRewrite(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Layout is unpackedOperands[operand][elem].
    SmallVector<SmallVector<Value>> unpackedOperands;
    for (auto operand : adaptor.getOperands()) {
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      unpackedOperands.push_back(subOperands);
    }

    int numElemsPerThread = getNumElementsPerThreads(op->getResult(0).getType(),
                                                     getTypeConverter());
    int packedElement = (int)op.getPackedElement();

    // These are checked by the verifier, so we don't need to raise a nice
    // error.
    assert(all_of(unpackedOperands, [&](auto &operands) {
      return (int)operands.size() == numElemsPerThread;
    }));
    if (numElemsPerThread % packedElement != 0) {
      // Pad with the undef for each operand to have a multiple of
      // packedElement elements.
      int numPaddedValue =
          packedElement - numElemsPerThread % packedElement;
      for (auto &operands : unpackedOperands) {
        operands.append(numPaddedValue, b.undef(operands[0].getType()));
      }
    }

    // Run the inline asm op on each block of elements.
    //
    // Layout is unpackedResults[result_idx][elem].
    //
    // This loop always runs at least once, even when the asm has no input
    // elements.
    SmallVector<SmallVector<Value>> unpackedResults(op->getNumResults());
    for (int i = 0; i < numElemsPerThread; i += packedElement) {
      // Block of elements to process with one call to the inline asm.  This is
      // ordered opposite `unpackedResults`: The outer dim is
      // packedElement, and the inner dim is the operand.
      SmallVector<SmallVector<Value>> block(packedElement);
      for (auto &os : unpackedOperands) {
        for (uint32_t j = 0; j < packedElement; j++) {
          block[j].push_back(os[i + j]);
        }
      }
      auto cur = createDestOps(op, adaptor, rewriter, block, loc);
      assert(cur.size() == unpackedResults.size());
      for (unsigned j = 0; j < cur.size(); j++) {
        unpackedResults[j].append(cur[j].begin(), cur[j].end());
      }
    }
    for (auto &results : unpackedResults) {
      results.resize(numElemsPerThread);
    }
    // Reorder and pack the results.
    SmallVector<Value> outs;
    for (size_t i = 0; i < unpackedResults.size(); i++) {
      outs.push_back(packLLElements(loc, getTypeConverter(), unpackedResults[i],
                                    rewriter, op->getResult(i).getType()));
    }

    rewriter.replaceOp(op, outs);
    return success();
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0][0],
                                         /*is_int_min_poison=*/false)};
  }
};

struct AbsFOpConversion
    : public ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return {b.and_(operands[0][0], maskConst)};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0][0])};
  }
};

struct SelectOpConversion
    : public ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SelectOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    SmallVector<Value, 3> llvmOperands;
    if (operands[0].size() == 2) {
      // Case of scalar condition with tensor operands.
      assert(op.getCondition().getType().isInteger(1));
      llvmOperands = {adaptor.getCondition(), operands[0][0], operands[0][1]};
    } else {
      llvmOperands = {operands[0][0], operands[0][1], operands[0][2]};
    }
    return {rewriter.create<LLVM::SelectOp>(
        loc, llvmOperands[1].getType(), llvmOperands,
        adaptor.getAttributes().getValue())};
  }
};
template <typename OpTy>
struct MinMaxFOpConversion
    : public ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>> {
  using Base = ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static_assert(std::is_same_v<OpTy, arith::MinimumFOp> ||
                    std::is_same_v<OpTy, arith::MaximumFOp>,
                "OpTy must be arith::MinimumFOp or arith::MaximumFOp");

  // Choose the destination op based on the OpTy.
  using DestOpNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinimumOp, LLVM::MaximumOp>::type;
  using DestOpNoNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinNumOp, LLVM::MaxNumOp>::type;

  explicit MinMaxFOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               bool hwNanPropagationSupported,
                               PatternBenefit benefit = 1)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        hwNanPropagationSupported(hwNanPropagationSupported) {}

  SmallVector<Value> createDestOps(OpTy op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (hwNanPropagationSupported) {
      return {rewriter.create<DestOpNanProp>(loc, elemTy, operands[0][0],
                                             operands[0][1])};
    }
    // Handle workaround for NaN propagation, i.e. software emulation of NaN
    // propagation. If any of the operands is NaN, return NaN.
    auto lhs = operands[0][0];
    auto rhs = operands[0][1];
    auto lhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, lhs, lhs);
    auto rhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, rhs, rhs);
    auto isNan = rewriter.create<LLVM::OrOp>(loc, lhsIsNan, rhsIsNan);
    auto nonNanRes = rewriter.create<DestOpNoNanProp>(loc, elemTy, lhs, rhs);

    auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);

    // Select the result based on the isNan flag.
    return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
  }

private:
  bool hwNanPropagationSupported;
};

struct ClampFOpConversion
    : public ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetInfo(targetInfo) {}

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Clip pattern not found, use min/max.
    if (op.getPropagateNan() == PropagateNan::ALL) {
      if (targetInfo.supportMaximumMinimum()) {
        auto v = rewriter.create<LLVM::MaximumOp>(loc, elemTy, operands[0][0],
                                                  operands[0][1]);
        return {rewriter.create<LLVM::MinimumOp>(loc, v, operands[0][2])};
      }
      // On pre-80 compute capability, we need to handle NaN propagation
      // manually. We need to check only the first operand for clamp.
      auto lhs = operands[0][0];
      auto isNan = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une,
                                                 lhs, lhs);
      auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                               operands[0][1]);
      auto nonNanRes = rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2]);
      auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);
      // Select the result based on the isNan flag.
      return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
    }

    // No NaN propagation.
    assert(op.getPropagateNan() == PropagateNan::NONE);
    auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                             operands[0][1]);
    return {rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2])};
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct MapElementwiseOpConversion
    : public ConvertOpToLLVMPattern<MapElementwiseOp> {
  using Base = ConvertOpToLLVMPattern<MapElementwiseOp>;
  using Adaptor = typename Base::OpAdaptor;

  using Base::Base;

  LogicalResult matchAndRewrite(MapElementwiseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();

    auto operands = adaptor.getOperands();
    const auto nOperands = operands.size();
    const auto nElems =
        cast<LLVM::LLVMStructType>(operands[0].getType()).getBody().size();
    const auto nElemsPerPack = op.getPack();
    if (nElems % nElemsPerPack != 0)
      return op->emitError()
             << "pack size must be a divisor of the number of elements per "
                "thread, but got pack = "
             << nElemsPerPack << ", elements per thread = " << nElems << "\n";

    const auto nPacks = nElems / nElemsPerPack;
    auto nArgsUnpacked = nElemsPerPack * nOperands;

    SmallVector<Value> scalarOperands(nOperands * nElems);
    for (auto iOp : llvm::seq(nOperands)) {
      auto elems = unpackLLElements(loc, operands[iOp], rewriter);
      assert(elems.size() == nElems);
      for (auto iPack : llvm::seq(nPacks)) {
        auto *packOperands =
            &scalarOperands[iPack * nArgsUnpacked + iOp * nElemsPerPack];
        auto *packElems = &elems[iPack * nElemsPerPack];
        for (auto iElem : llvm::seq(nElemsPerPack)) {
          packOperands[iElem] = packElems[iElem];
        }
      }
    }

    auto &scalarOp = op.getScalarOp();
    Region &parent = *rewriter.getBlock()->getParent();

    auto nOutputs = op.getNumResults();
    SmallVector<Value> scalarOutputs(nOutputs * nElems);
    for (auto iPack : llvm::seq(nPacks)) {
      ArrayRef<Value> packedArgs(&scalarOperands[iPack * nArgsUnpacked],
                                 nArgsUnpacked);
      auto packResults = inlineRegion<triton::MapElementwiseReturnOp>(
          rewriter, scalarOp, packedArgs, loc);
      assert(packResults.size() == nOutputs * nElemsPerPack);
      for (auto iOut : llvm::seq(nOutputs)) {
        auto *packOutputs =
            &scalarOutputs[iOut * nElems + iPack * nElemsPerPack];
        for (auto iElem : llvm::seq(nElemsPerPack)) {
          packOutputs[iElem] = packResults[iOut * nElemsPerPack + iElem];
        }
      }
    }

    SmallVector<Value> packedOutputs(nOutputs);
    for (auto iOut : llvm::seq(nOutputs)) {
      ArrayRef<Value> vals(&scalarOutputs[iOut * nElems], nElems);
      packedOutputs[iOut] =
          packLLElements(loc, typeConverter, vals, rewriter, op.getType(iOut));
    }
    rewriter.replaceOp(op, packedOutputs);
    return success();
  }
};

} // namespace

void mlir::triton::ascend::populateAscendElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  // these patterns are copied over from triton
  // ideally if we could modify the triton file we can reuse these
#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::TruncFOp, LLVM::FPTruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::ExtFOp, LLVM::FPExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::FPToSIOp, LLVM::FPToSIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(math::TanhOp, ascend_dpx::TanhOp)
  POPULATE_UNARY_OP(math::FloorOp, ascend_dpx::FloorOp)
  POPULATE_UNARY_OP(math::CeilOp, ascend_dpx::CeilOp)
  POPULATE_UNARY_OP(math::LogOp, ascend_dpx::LogOp)
  POPULATE_UNARY_OP(math::Log2Op, ascend_dpx::Log2Op)
  POPULATE_UNARY_OP(math::CosOp, ascend_dpx::CosOp)
  POPULATE_UNARY_OP(math::SinOp, ascend_dpx::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, ascend_dpx::SqrtOp)
  POPULATE_UNARY_OP(math::RsqrtOp, ascend_dpx::RSqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, ascend_dpx::ExpOp)
  POPULATE_UNARY_OP(math::Exp2Op, ascend_dpx::Exp2Op)
  POPULATE_UNARY_OP(math::ErfOp, ascend_dpx::ErfOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
  POPULATE_UNARY_OP(triton::PreciseSqrtOp, ascend_dpx::SqrtOp)
#undef POPULATE_UNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::SubFOp, LLVM::FSubOp)
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::AddFOp, LLVM::FAddOp)
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::MulFOp, LLVM::FMulOp)
  POPULATE_BINARY_OP(arith::DivSIOp, ascend_dpx::DivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, ascend_dpx::DivOp)
  POPULATE_BINARY_OP(arith::DivFOp, ascend_dpx::DivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
  // fmin (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MinNumFOp, LLVM::MinNumOp)
  // fmax (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MaxNumFOp, LLVM::MaxNumOp)
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp) // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp) // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp) // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp) // umax
  POPULATE_BINARY_OP(triton::MulhiUIOp, ascend_dpx::UmulhiOp)
  POPULATE_BINARY_OP(triton::PreciseDivFOp, ascend_dpx::DivOp)
#undef POPULATE_BINARY_OP

  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, false, benefit);
  // hwNanPropagationSupported? assumed to be false
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  patterns.add<ElementwiseOpConversion<math::FmaOp, LLVM::FMAOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter, benefit);
  patterns.add<AbsIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SelectOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MapElementwiseOpConversion>(typeConverter, benefit);

  // custom pattern
  patterns.add<AscendSIToFPOpConversion>(typeConverter, axisInfoAnalysis,
                                         benefit);
}
