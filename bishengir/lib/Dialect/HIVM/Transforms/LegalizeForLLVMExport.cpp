//===- LegalizeForLLVMExport.cpp - Prepare HIVM for LLVM translation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Transforms.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>

using namespace mlir;
using namespace mlir::hivm;

using GetBlockIdxLowering =
    OneToOneConvertToLLVMPattern<GetBlockIdxOp, GetBlockIdxInstrOp>;

using GetBlockNumLowering =
    OneToOneConvertToLLVMPattern<GetBlockNumOp, GetBlockNumInstrOp>;

using GetSubBlockIdxLowering =
    OneToOneConvertToLLVMPattern<GetSubBlockIdxOp, GetSubBlockIdxInstrOp>;

using GetSubBlockNumLowering =
    OneToOneConvertToLLVMPattern<GetSubBlockNumOp, GetSubBlockNumInstrOp>;

template <typename Op>
LogicalResult convertToImmInstrOp(Op convertOp) {
  if (convertOp.getStaticEventId().has_value()) {
    return success();
  }

  if (convertOp.getDynamicEventId().getDefiningOp() &&
      dyn_cast<arith::ConstantOp>(
          convertOp.getDynamicEventId().getDefiningOp())) {
    return success();
  }

  return failure();
}

template <typename Op>
LogicalResult convertToRegInstrOp(Op convertOp) {
  if (convertOp.getStaticEventId().has_value()) {
    return failure();
  }

  if (convertOp.getDynamicEventId().getDefiningOp() &&
      dyn_cast<arith::ConstantOp>(
          convertOp.getDynamicEventId().getDefiningOp())) {
    return failure();
  }

  return success();
}

template <typename Op, typename ImmT, typename RegT>
struct HIVMSetWaitFlagOpLowering : public ConvertOpToLLVMPattern<Op> {
  explicit HIVMSetWaitFlagOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<Op>(converter) {}

  LogicalResult
  matchAndRewrite(Op convertOp, typename Op::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // check valid operand
    auto loc = convertOp.getLoc();
    uint64_t setPipe = static_cast<uint64_t>(convertOp.getSetPipe().getPipe());
    uint64_t waitPipe =
        static_cast<uint64_t>(convertOp.getWaitPipe().getPipe());

    // convert SetFlagOp/WaitFlagOp to hivm.SET.FLAG.IMM/hivm.WAIT.FLAG.IMM when
    // eventId is an InterAttr or a value of arith.constant.
    if (succeeded(convertToImmInstrOp<Op>(convertOp))) {
      uint64_t eventId;
      if (convertOp.getStaticEventId().has_value()) {
        eventId =
            static_cast<uint64_t>(convertOp.getStaticEventId()->getEvent());
      } else {
        // convertToImmInstrOp guarantees event id is a constant op in this case
        eventId = static_cast<uint64_t>(cast<arith::ConstantIntOp>(
                      convertOp.getDynamicEventId().getDefiningOp())
                      .value());
      }
      auto result = rewriter.create<ImmT>(loc, setPipe, waitPipe, eventId);
      rewriter.replaceOp(convertOp, result);
      return success();
    }

    // convert SetFlagOp/WaitFlagOp to hivm.SET.FLAG.REG/hivm.WAIT.FLAG.REG when
    // eventId is a variable.
    if (succeeded(convertToRegInstrOp<Op>(convertOp))) {
      Value eventId = convertOp.getDynamicEventId();
      auto result = rewriter.create<RegT>(loc, setPipe, waitPipe, eventId);
      rewriter.replaceOp(convertOp, result);
      return success();
    }

    return failure();
  }
};

struct HIVMPipeBarrierOpLowering
    : public ConvertOpToLLVMPattern<PipeBarrierOp> {
  explicit HIVMPipeBarrierOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<PipeBarrierOp>(converter) {}
  LogicalResult
  matchAndRewrite(PipeBarrierOp convertOp, PipeBarrierOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = convertOp.getLoc();
    uint64_t pipebarrier = static_cast<uint64_t>(convertOp.getPipe().getPipe());
    auto result = rewriter.create<PipeBarrierInstrOp>(loc, pipebarrier);

    rewriter.replaceOp(convertOp, result);
    return success();
  }
};

inline Value GetBlockSyncInstrConfig(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const hivm::SyncBlockInstrMode mode,
                                     OpFoldResult flagID) {
  assert(mode >= hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION &&
         mode <= hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION &&
         "BlockSyncInstrMode is illegal");
  Value result;
  // xd[15:8] contains mode, xd[7:0] contains flagID
  if (auto attr = dyn_cast_if_present<Attribute>(flagID)) {
    auto eventAttr = llvm::cast<IntegerAttr>(attr);
#ifndef NDEBUG
    const int flagIdUpperBound = 1 << 8;
    assert(eventAttr.getInt() >= 0 && eventAttr.getInt() < flagIdUpperBound &&
           "FlagID is invalid");
#endif
    auto config = static_cast<uint64_t>(
        (0x0001 | ((static_cast<uint32_t>(mode) & 0x0f) << 4) |
         ((static_cast<uint64_t>(eventAttr.getInt()) & 0x0f) << 8)));
    auto i64Ty = rewriter.getI64Type();
    result = rewriter.create<LLVM::ConstantOp>(
        loc, i64Ty, rewriter.getIntegerAttr(i64Ty, config));
  } else {
    auto flagValue = llvm::dyn_cast<Value>(flagID);
    assert(flagValue);
    uint64_t staticPart = static_cast<uint64_t>(
        0x0001 | ((static_cast<uint32_t>(mode) & 0x0f) << 4));
    Value staticValue = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, rewriter.getI64Type(), staticPart);
    Value shift8 = rewriter.create<mlir::arith::ConstantIntOp>(
        loc, rewriter.getI64Type(), 8);
    Value shiftFlagVal =
        rewriter.create<mlir::arith::ShLIOp>(loc, flagValue, shift8);
    result =
        rewriter.create<mlir::arith::OrIOp>(loc, staticValue, shiftFlagVal);
  }
  return result;
}

template <typename Op, typename ImmT, typename RegT>
struct LowerBlockSetWaitToIntraBlockSetWait {
    static Value getPairFlagId(Value originalFlagId, PatternRewriter &rewriter, Location loc) {
      Value offsetConstant = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(util::INTRA_BLOCK_FLAG_ID_OFFSET));
      Value newValue = rewriter.create<LLVM::AddOp>(loc, originalFlagId, offsetConstant);
      return newValue;
    }
    static LogicalResult Lower(Op convertOp, ConversionPatternRewriter &rewriter) {
      std::optional<uint64_t> flagId;
      auto loc = convertOp.getLoc();
      auto core = convertOp.getTcoreTypeAttr().getTcoretype();
      uint64_t pipeVal;
      if (isa<SyncBlockSetOp>(convertOp)) {
        pipeVal = static_cast<uint64_t>(convertOp.getTpipeAttr().getPipe());
      } else {
        pipeVal = static_cast<uint64_t>(convertOp.getPipeAttr().getPipe());
      }
      if (convertOp.getStaticFlagId().has_value()) {
        flagId = static_cast<uint64_t>(convertOp.getStaticFlagId()->getInt());
      } else if (auto constOp = dyn_cast_if_present<arith::ConstantIntOp>(
                  convertOp.getDynamicFlagId().getDefiningOp())) {
        flagId = static_cast<uint64_t>(constOp.value());
      }
      if (flagId.has_value()) {
        if (core == TCoreType::CUBE) {
          rewriter.create<ImmT>(loc, pipeVal, flagId.value() + util::INTRA_BLOCK_FLAG_ID_OFFSET);
        }
        auto result = rewriter.create<ImmT>(loc, pipeVal, flagId.value());
        rewriter.replaceOp(convertOp, result);
        return success();
      }
      if (auto syncIdVal = convertOp.getDynamicFlagId()) {
        if (core == TCoreType::CUBE) {
          rewriter.create<RegT>(loc, pipeVal, getPairFlagId(syncIdVal, rewriter, loc));
        }
        auto result = rewriter.create<RegT>(loc, pipeVal, syncIdVal);
        rewriter.replaceOp(convertOp, result);
        return success();
      }
      return failure();
    }
};

using LowerBlockSetToIntraBlockSet =
    LowerBlockSetWaitToIntraBlockSetWait<SyncBlockSetOp, SetIntraBlockImmInstrOp,
                                             SetIntraBlockRegInstrOp>;

using LowerBlockWaitToIntraBlockSet =
    LowerBlockSetWaitToIntraBlockSetWait<SyncBlockWaitOp, WaitIntraBlockImmInstrOp,
                                             WaitIntraBlockRegInstrOp>;
struct HIVMSetBlockSyncOpLowering
    : public ConvertOpToLLVMPattern<SyncBlockSetOp> {
  bool isRegBased;
  explicit HIVMSetBlockSyncOpLowering(LLVMTypeConverter &converter, bool isRegBased)
      : ConvertOpToLLVMPattern<SyncBlockSetOp>(converter),
        isRegBased(isRegBased) {}
  LogicalResult
  matchAndRewrite(SyncBlockSetOp convertOp, SyncBlockSetOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isRegBased && 
      convertOp.getTsyncInstrMode().getSyncInstrMode() == SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION) {
      return LowerBlockSetToIntraBlockSet::Lower(convertOp, rewriter);
    }
    auto loc = convertOp.getLoc();
    auto fftsBaseAddr = convertOp.getFftsBaseAddr();
    if (fftsBaseAddr) {
      rewriter.create<SetFftsBaseAddrInstrOp>(loc, fftsBaseAddr);
    }
    auto configVal = GetBlockSyncInstrConfig(
        rewriter, loc, convertOp.getTsyncInstrMode().getSyncInstrMode(),
        convertOp.getFlagId());
    uint64_t pipeVal =
        static_cast<uint64_t>(convertOp.getTpipeAttr().getPipe());
    auto result = rewriter.create<SetCrossCoreInstrOp>(loc, pipeVal, configVal);
    rewriter.replaceOp(convertOp, result);
    return success();
  }
};

struct HIVMWaitBlockSyncOpLowering
    : public ConvertOpToLLVMPattern<SyncBlockWaitOp> {
  explicit HIVMWaitBlockSyncOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<SyncBlockWaitOp>(converter) {}
  LogicalResult
  matchAndRewrite(SyncBlockWaitOp convertOp, SyncBlockWaitOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i64Ty = rewriter.getI64Type();
    auto loc = convertOp.getLoc();
    Value eventID;
    auto blockSyncID = convertOp.getFlagId();
    if (auto attr = dyn_cast_if_present<Attribute>(blockSyncID)) {
      auto eventAttr = llvm::cast<IntegerAttr>(attr);
      eventID = rewriter.create<LLVM::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, eventAttr.getInt()));
    } else {
      auto val = llvm::dyn_cast<Value>(blockSyncID);
      assert(val);
      eventID = val;
    }
    auto result = rewriter.create<WaitFlagDevInstrOp>(loc, eventID);
    rewriter.replaceOp(convertOp, result);
    return success();
  }
};

struct HIVMWaitBlockSyncPipeOpLowering
    : public ConvertOpToLLVMPattern<SyncBlockWaitOp> {
  explicit HIVMWaitBlockSyncPipeOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<SyncBlockWaitOp>(converter) {}
  LogicalResult
  matchAndRewrite(SyncBlockWaitOp convertOp, SyncBlockWaitOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (convertOp.getTsyncInstrMode().getSyncInstrMode() == SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION) {
      return LowerBlockWaitToIntraBlockSet::Lower(convertOp, rewriter);
    }
    auto loc = convertOp.getLoc();
    uint64_t pipeVal =
        static_cast<uint64_t>(convertOp.getPipeAttr().getPipe());
    std::optional<uint64_t> flagId;
    if (convertOp.getStaticFlagId().has_value()) {
      flagId = static_cast<uint64_t>(convertOp.getStaticFlagId()->getInt());
    } else if (auto constOp = dyn_cast_if_present<arith::ConstantIntOp>(
                   convertOp.getDynamicFlagId().getDefiningOp())) {
      flagId = static_cast<uint64_t>(constOp.value());
    }
    if (flagId.has_value()) {
      // convert SyncBlockWaitOp to hivm.WAIT.FLAG.DEV.PIPE.IMM when eventId
      // is an InterAttr or a result of arith.constant.
      auto result = rewriter.create<WaitFlagDevPipeImmInstrOp>(loc, pipeVal,
                                                               flagId.value());
      rewriter.replaceOp(convertOp, result);
      return success();
    } else if (auto flagIdVal = convertOp.getDynamicFlagId()) {
      // convert SyncBlockWaitOp to hivm.WAIT.FLAG.DEV.PIPE.REG when eventId
      // is a variable.
      auto result =
          rewriter.create<WaitFlagDevPipeRegInstrOp>(loc, pipeVal, flagIdVal);
      rewriter.replaceOp(convertOp, result);
      return success();
    }
    return failure();
  }
};

struct HIVMSetFFTSBaseAddrOpLowering
    : public ConvertOpToLLVMPattern<SetFFTSBaseAddrOp> {
  explicit HIVMSetFFTSBaseAddrOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<SetFFTSBaseAddrOp>(converter) {}
  LogicalResult
  matchAndRewrite(SetFFTSBaseAddrOp convertOp, SetFFTSBaseAddrOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = convertOp.getLoc();
    auto fftsBaseAddr = convertOp.getFftsBaseAddr();

    rewriter.replaceOp(
        convertOp, rewriter.create<SetFftsBaseAddrInstrOp>(loc, fftsBaseAddr));
    return success();
  }
};


struct HIVMSetMaskNormOpLowering
    : public ConvertOpToLLVMPattern<SetMaskNormOp> {
  explicit HIVMSetMaskNormOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<SetMaskNormOp>(converter) {}
  LogicalResult
  matchAndRewrite(SetMaskNormOp setMaskNormOp, SetMaskNormOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = setMaskNormOp.getLoc();
    Type i64 = rewriter.getI64Type();
    Value ctrl = rewriter.create<GetCtrlInstrOp>(loc, i64);
    Value cst = rewriter.create<LLVM::ConstantOp>(loc, i64, MaskControlBit);
    Value sVal = rewriter.create<SBitSet0InstrOp>(loc, i64, ctrl, cst);
    rewriter.replaceOp(setMaskNormOp,
                       rewriter.create<SetCtrlInstrOp>(loc, sVal));
    return success();
  }
};

struct HIVMSetCtrlOpLowering : public ConvertOpToLLVMPattern<SetCtrlOp> {
  explicit HIVMSetCtrlOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<SetCtrlOp>(converter) {}
  LogicalResult
  matchAndRewrite(SetCtrlOp setCtrlOp, SetCtrlOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = setCtrlOp.getLoc();
    Type i64 = rewriter.getI64Type();
    Value ctrl = rewriter.create<GetCtrlInstrOp>(loc, i64);
    Value idx = rewriter.create<LLVM::ConstantOp>(loc, i64, setCtrlOp.getIdx());
    Value sVal;
    if (setCtrlOp.getEnable()) {
      sVal = rewriter.create<SBitSet1InstrOp>(loc, i64, ctrl, idx);
    } else {
      sVal = rewriter.create<SBitSet0InstrOp>(loc, i64, ctrl, idx);
    }
    rewriter.replaceOp(setCtrlOp, rewriter.create<SetCtrlInstrOp>(loc, sVal));
    return success();
  }
};

struct HIVMDCCIDstOpLowering : public ConvertOpToLLVMPattern<DCCIOp> {
  explicit HIVMDCCIDstOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<DCCIOp>(converter) {}
  Value createBufferPtr(ConversionPatternRewriter &rewriter, Location loc,
                        Value llvmStructVal, Value memrefVal) const {
    Value base;
    if (memrefVal != nullptr) {
      MemRefDescriptor memRefDescriptor((llvmStructVal));
      auto origMemrefType = dyn_cast<MemRefType>((memrefVal).getType());
      base = memRefDescriptor.bufferPtr(rewriter, loc, *getTypeConverter(),
                                        origMemrefType);
    } else {
      unsigned addrSpaceInt =
          static_cast<unsigned>(mlir::hivm::AddressSpace::GM);
      auto nullMemrefType =
          MemRefType::get({}, rewriter.getI64Type(), AffineMap{},
                          rewriter.getI64IntegerAttr(addrSpaceInt));
      Type pType =
          LLVM::LLVMPointerType::get(rewriter.getContext(), addrSpaceInt);
      Value baseCst =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
      Value allocatedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, pType, baseCst);
      MemRefDescriptor descriptor = createMemRefDescriptor(
          loc, nullMemrefType, allocatedPtr, allocatedPtr, {}, {}, rewriter);
      base = descriptor.bufferPtr(rewriter, loc, *getTypeConverter(),
                                  nullMemrefType);
    }
    return base;
  }

  LogicalResult
  matchAndRewrite(DCCIOp dcciOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = dcciOp->getLoc();
    auto dcciMode = dcciOp.getMode();
    auto dataCacheKind = dcciOp.getDataCacheKind();
    Value dcciModeVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), static_cast<int64_t>(dcciMode));
    Value dataCacheKindVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), static_cast<int64_t>(dataCacheKind));

    auto memrefVal = dcciOp.getPtr();

    auto hivmAddrSpace = mlir::hivm::AddressSpace::GM;
    if (memrefVal) {
      hivmAddrSpace = getHIVMAddressSpace(memrefVal.getType());
    }

    Value base = createBufferPtr(rewriter, loc, adaptor.getPtr(), memrefVal);

    switch (hivmAddrSpace) {
    case mlir::hivm::AddressSpace::UB:
      rewriter.replaceOp(dcciOp, rewriter.create<DCCIDstUBInstrOp>(
                                     loc, base, dcciModeVal, dataCacheKindVal));
      break;
    case mlir::hivm::AddressSpace::GM:
      rewriter.replaceOp(dcciOp, rewriter.create<DCCIDstInstrOp>(
                                     loc, base, dcciModeVal, dataCacheKindVal));
      break;
    default:
      llvm_unreachable("should not expect address space other than UB and GM");
      break;
    }
    return success();
  }
};

using HIVMSetFlagOpLowering =
    HIVMSetWaitFlagOpLowering<SetFlagOp, SetFlagImmInstrOp, SetFlagRegInstrOp>;
using HIVMWaitFlagOpLowering =
    HIVMSetWaitFlagOpLowering<WaitFlagOp, WaitFlagImmInstrOp,
                              WaitFlagRegInstrOp>;

/// Populate the given list with patterns that convert from HIVM to LLVM.
void mlir::populateHIVMLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool isRegBased) {
  // Populate conversion patterns
  // clang-format off
  patterns.add<GetBlockIdxLowering, GetBlockNumLowering,
               GetSubBlockIdxLowering, GetSubBlockNumLowering,
               HIVMSetFlagOpLowering,
               HIVMWaitFlagOpLowering,
               HIVMPipeBarrierOpLowering,
               HIVMSetFFTSBaseAddrOpLowering,
               HIVMSetMaskNormOpLowering,
               HIVMSetCtrlOpLowering,
               HIVMDCCIDstOpLowering
  >(converter);
  patterns.add<HIVMSetBlockSyncOpLowering>(converter, isRegBased);
  // clang-format on
  if (isRegBased) {
    patterns.add<HIVMWaitBlockSyncPipeOpLowering>(converter);
  } else {
    patterns.add<HIVMWaitBlockSyncOpLowering>(converter);
  }
}

void mlir::configureHIVMLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<GetBlockIdxInstrOp>();
  target.addLegalOp<GetBlockNumInstrOp>();
  target.addLegalOp<GetSubBlockIdxInstrOp>();
  target.addLegalOp<GetSubBlockNumInstrOp>();
  target.addLegalOp<SetFlagImmInstrOp>();
  target.addLegalOp<WaitFlagImmInstrOp>();
  target.addLegalOp<SetFlagRegInstrOp>();
  target.addLegalOp<WaitFlagRegInstrOp>();
  target.addLegalOp<PipeBarrierInstrOp>();
  target.addLegalOp<SetCrossCoreInstrOp>();
  target.addLegalOp<SetFftsBaseAddrInstrOp>();
  target.addLegalOp<WaitFlagDevInstrOp>();
  target.addLegalOp<WaitFlagDevPipeImmInstrOp>();
  target.addLegalOp<WaitFlagDevPipeRegInstrOp>();
  target.addLegalOp<SetMaskNormInstrOp>();
  target.addLegalOp<GetCtrlInstrOp>();
  target.addLegalOp<SBitSet0InstrOp>();
  target.addLegalOp<SBitSet1InstrOp>();
  target.addLegalOp<SetCtrlInstrOp>();
  target.addLegalOp<DCCIDstInstrOp>();
  target.addLegalOp<DCCIDstUBInstrOp>();
  target.addLegalOp<SetIntraBlockRegInstrOp>();
  target.addLegalOp<SetIntraBlockImmInstrOp>();
  target.addLegalOp<WaitIntraBlockRegInstrOp>();
  target.addLegalOp<WaitIntraBlockImmInstrOp>();
  // clang-format on
}
