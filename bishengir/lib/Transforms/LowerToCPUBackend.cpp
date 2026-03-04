//===--------------------- LowerToCPUBackend.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

namespace bishengir {
#define GEN_PASS_DEF_LOWERTOCPUBACKEND
#include "bishengir/Transforms/Passes.h.inc"
} // namespace bishengir

using namespace mlir;

namespace {

template <typename T>
static void eraseMemRefTypeMemSpace(T &input) {
  Type originType;
  if constexpr (std::is_same_v<T, Value>)
    originType = input.getType();
  if constexpr (std::is_same_v<T, Type>)
    originType = input;

  if (auto baseMemrefType = llvm::dyn_cast<BaseMemRefType>(originType)) {
    BaseMemRefType erasedType;
    if (auto memRefType = llvm::dyn_cast<MemRefType>(baseMemrefType)) {
      erasedType =
          MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                          memRefType.getLayout());
    } else if (auto unrankedMemRefType =
                   llvm::dyn_cast<UnrankedMemRefType>(baseMemrefType)) {
      erasedType =
          UnrankedMemRefType::get(unrankedMemRefType.getElementType(), {});
    } else {
      llvm_unreachable("Unexpected BaseMemRefType");
    }

    if constexpr (std::is_same_v<T, Value>)
      input.setType(erasedType);
    if constexpr (std::is_same_v<T, Type>)
      input = erasedType;
  }
}

// Thanks mlir toy example
/// Create a function declaration for printf, the signature is:
///   * `i32 (i8*, ...)`
static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                /*isVarArg=*/true);
  return llvmFnType;
}

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                    getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     std::string name, StringRef value,
                                     ModuleOp module, bool uniqueFlag) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global = module.lookupSymbol<LLVM::GlobalOp>(name);
  if (uniqueFlag || !global) {
    while (global) {
      name = "intrinsic" +
             std::to_string(static_cast<uint64_t>(llvm::hash_value(name)));
      global = module.lookupSymbol<LLVM::GlobalOp>(name);
    }

    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

// Use llvm dialect to express ``printf`` function directly
template <typename HIVMOp>
class HIVMIntrinsicOpPrint : public OpRewritePattern<HIVMOp> {
  using OpRewritePattern<HIVMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HIVMOp op,
                                PatternRewriter &rewriter) const final {
    auto module = op->template getParentOfType<mlir::ModuleOp>();
    auto *context = rewriter.getContext();
    auto printfDeclRef = getOrInsertPrintf(rewriter, module);

    std::string stringContent;
    llvm::raw_string_ostream stringContentStream(stringContent);
    op->print(stringContentStream);
    // There exist double formatSpecifier right now, including '%s' and '%s%lld'
    // For SetFlagOp & WaitFlagOp, it should consider both as event id may be
    // dynamic value optionally.
    llvm::SmallVector<Value> printfFuncArgs;

    // ToDo: Generalize following logic to support any HIVM intrinsic op with
    // any operands
    if (op->getNumOperands() == 0) {
      Value formatSpecifier =
          getOrCreateGlobalString(op.getLoc(), rewriter, "frmt_spec_s",
                                  StringRef("%s\n\0", 4), module, false);
      printfFuncArgs.push_back(formatSpecifier);
    } else {
      Value formatSpecifier =
          getOrCreateGlobalString(op.getLoc(), rewriter, "frmt_spec_sAndI",
                                  StringRef("%s %lld\n\0", 9), module, false);
      printfFuncArgs.push_back(formatSpecifier);
    }
    std::string stringName =
        "intrinsic" +
        std::to_string(static_cast<uint64_t>(hash_value(op->getName())));
    // To follow C style format, here needs to carry std::string end char `\0`
    const char *contentCharArry = stringContent.c_str();
    Value stringArg = getOrCreateGlobalString(
        op.getLoc(), rewriter, stringName,
        StringRef(contentCharArry, stringContent.size() + 1), module, true);
    printfFuncArgs.push_back(stringArg);
    if (op->getNumOperands() != 0)
      printfFuncArgs.push_back(op->getOperand(0));

    rewriter.create<LLVM::CallOp>(op.getLoc(), getPrintfType(context),
                                  printfDeclRef,
                                  ArrayRef<Value>(printfFuncArgs));
    rewriter.eraseOp(op);
    return success();
  }
};

// hivm sepcial intrinsic op
static void populateHIVMIntrinsicPrintPatterns(RewritePatternSet &patterns) {
  patterns.add<HIVMIntrinsicOpPrint<hivm::SetFlagOp>,
               HIVMIntrinsicOpPrint<hivm::WaitFlagOp>,
               HIVMIntrinsicOpPrint<hivm::PipeBarrierOp>,
               HIVMIntrinsicOpPrint<hivm::SetMaskNormOp>>(
      patterns.getContext());
}

class PointerCastToAllocaPattern
    : public OpRewritePattern<hivm::PointerCastOp> {
  using OpRewritePattern<hivm::PointerCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(hivm::PointerCastOp op,
                                PatternRewriter &rewriter) const final {
    // Convert hivm::PointerCastOp to memref::AllocaOp which would lower to
    // simple stack space allocation in CPU backend.
    rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        op, op.getResult().getType(),
        /*alignment 32 bits*/ rewriter.getI64IntegerAttr(32));
    return success();
  }
};

class LaunchBlockToLoopPattern : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  explicit LaunchBlockToLoopPattern(MLIRContext *context,
                                    bool enableTritonKernelCompile)
      : OpRewritePattern<func::FuncOp>(context) {
    enableTritonKernelCompile_ = enableTritonKernelCompile;
  }

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const final {
    auto loc = funcOp.getLoc();
    Region &funcRegion = funcOp.getBody();
    if (!funcRegion.hasOneBlock())
      return rewriter.notifyMatchFailure(
          funcOp, "just support simple function with single block");

    assert(funcRegion.front().mightHaveTerminator() &&
           "function in this period must have return statement with no value");

    // Restore origin op in func block, which will be moved to scf::ForOp body
    llvm::SmallVector<Operation *> originFuncOp;
    for (Operation &op : funcRegion.front().without_terminator())
      originFuncOp.push_back(&op);

    // GetBlockIdxOp shouldn't be nested block of non-funcOp.
    if (!llvm::any_of(originFuncOp, [](Operation *op) {
          return llvm::isa<hivm::GetBlockIdxOp>(op);
        }))
      return rewriter.notifyMatchFailure(
          funcOp, "current func body doesn't contain hivm::GetBlockIdxOp");

    // Create scf::for loop to represent multiple blocks launch
    rewriter.setInsertionPointToStart(&funcRegion.front());
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value upperBound;
    if (auto blockDimAttr = funcOp->getAttr(hacc::BlockDimAttr::name)) {
      auto blockDimIntAttr = llvm::cast<IntegerAttr>(blockDimAttr);
      upperBound = rewriter.create<arith::ConstantIndexOp>(
          loc, blockDimIntAttr.getInt());
    } else if (enableTritonKernelCompile_) {
      Value intermediate = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      constexpr int gridArgNum = 3;
      auto args = funcOp.getArguments();
      assert(funcOp.getNumArguments() >= gridArgNum &&
             "There must have 3 arguments at least when exists use "
             "of block idx in triton compilation");

      for (int i = 0; i < gridArgNum; ++i)
        intermediate = rewriter.create<arith::MulIOp>(
            loc, intermediate, *(args.end() - gridArgNum + i));
      upperBound = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), intermediate);
    } else {
      llvm_unreachable("There's no other state to carry block dim info except "
                       "block dim attr or triton arguments");
    }

    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
    auto &forBlock = forOp.getRegion().front();
    assert(forBlock.getOperations().size() == 1 &&
           "Initialized scf::for should just have only one scf::yield op");

    auto yieldIterator = forBlock.begin();
    rewriter.setInsertionPointToStart(&forBlock);
    for (Operation *op : originFuncOp) {
      if (auto getBlockIdxOp = llvm::dyn_cast<hivm::GetBlockIdxOp>(op)) {
        assert(forOp.getSingleInductionVar() == nullptr &&
               "Initialized scf::for should just have only one scf::yield op");
        auto currentIdx = forOp.getSingleInductionVar().value();
        auto blockIdxSubstitute = rewriter.create<arith::IndexCastOp>(
            op->getLoc(), rewriter.getI64Type(), currentIdx);

        rewriter.replaceOp(getBlockIdxOp, blockIdxSubstitute);
        op = blockIdxSubstitute;
      }
      op->moveBefore(&forBlock, yieldIterator);
    }

    return success();
  }

private:
  bool enableTritonKernelCompile_{false};
};

static void populateHIVMToCPUPatterns(RewritePatternSet &patterns,
                                      bool enableTritonKernelCompile) {
  patterns.add<PointerCastToAllocaPattern>(patterns.getContext());
  patterns.add<LaunchBlockToLoopPattern>(patterns.getContext(),
                                         enableTritonKernelCompile);
}

} // namespace

class LowerToCPUBackendPass
    : public bishengir::impl::LowerToCPUBackendBase<LowerToCPUBackendPass> {
public:
  explicit LowerToCPUBackendPass(
      const bishengir::LowerToCPUBackendOptions &options)
      : LowerToCPUBackendBase(options) {}
  void runOnOperation() override;
};

void LowerToCPUBackendPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Eliminate memory space attribute of memref type
  module.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      eraseMemRefTypeMemSpace<Value>(operand);

    // For op with results, just reset result value type
    for (Value result : op->getResults())
      eraseMemRefTypeMemSpace<Value>(result);

    // For op which holds regions, just reset block arguments of memref type
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          eraseMemRefTypeMemSpace<Value>(arg);
        }
      }
    }

    // Adjust func type
    if (auto funcOp = llvm::dyn_cast<func::FuncOp>(op)) {
      assert(op->getNumResults() == 0 && "non-externl function shouldn't have "
                                         "return value after bufferization!");

      llvm::SmallVector<Type> argTypes(funcOp.getArgumentTypes());
      for (auto &argType : argTypes)
        eraseMemRefTypeMemSpace<Type>(argType);

      auto newFuncType =
          funcOp.getFunctionType().clone(argTypes, funcOp.getResultTypes());
      funcOp.setFunctionType(newFuncType);
    }
  });

  RewritePatternSet patterns(&getContext());
  populateHIVMToCPUPatterns(patterns, enableTritonKernelCompile);
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }

  // Here mark all HIVM specially explicit intrinsic
  LLVMConversionTarget target(getContext());
  target.addIllegalOp<hivm::SetFlagOp, hivm::WaitFlagOp, hivm::PipeBarrierOp,
                      hivm::SetMaskNormOp,
                      // TODO: next intrinsics haven't been supported
                      hivm::SetFFTSBaseAddrOp, hivm::DebugOp, hivm::InitDebugOp,
                      hivm::FinishDebugOp, hivm::SyncBlockOp,
                      hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>();
  RewritePatternSet printPatterns(&getContext());
  populateHIVMIntrinsicPrintPatterns(printPatterns);
  if (failed(applyPartialConversion(module, target, std::move(printPatterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> bishengir::createLowerToCPUBackendPass(
    const bishengir::LowerToCPUBackendOptions &options) {
  return std::make_unique<LowerToCPUBackendPass>(options);
}