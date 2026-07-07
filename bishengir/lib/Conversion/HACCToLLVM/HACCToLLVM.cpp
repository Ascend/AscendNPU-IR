//===- HACCToLLVM.cpp - HACC to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert HACC dialect into the
// LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/HACCToLLVM/HACCToLLVM.h"
#include "bishengir/Conversion/HACCToLLVM/ConversionUtils.h"
#include "bishengir/Conversion/HACCToLLVM/GetOrCreateUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <utility>

namespace mlir {
#define GEN_PASS_DEF_CONVERTHACCTOLLVM
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hacc;
using namespace mlir::utils::debugger;

#define PASS_NAME "convert-hacc-to-llvm"
#define DEBUG_TYPE "convert-hacc-to-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define LLDBG(X)                                                               \
  LLVM_DEBUG(DBGS() << __FILE__ << ":" << __LINE__ << " " << X << "\n")

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

bool isCallingDevice(LLVM::LLVMFuncOp hostFuncOp) {
  bool isCallingDevice = false;
  hostFuncOp.walk([&](LLVM::CallOp callOp) {
    auto calleeOp =
        mlir::utils::getCalledFunction<hacc::HACCFunction, LLVM::CallOp>(
            callOp);
    if (!calleeOp)
      return WalkResult::interrupt();
    if (calleeOp.isDevice()) {
      isCallingDevice = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return isCallingDevice;
}

/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertHACCToLLVM
    : public impl::ConvertHACCToLLVMBase<ConvertHACCToLLVM> {
  using Base::Base;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    LowerToLLVMOptions lowerOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), lowerOptions);
    // This will run hacc on host function
    target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
      if (!hacc::utils::isHost(op))
        return true;

      return !isCallingDevice(op);
    });
    RewritePatternSet patterns(&getContext());
    mlir::hacc::populateHACCToLLVMConversionPatterns(typeConverter, patterns,
                                                     tempDeviceLLVMFilePath);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Prepare HACC
//===----------------------------------------------------------------------===//

void registerKernel(ModuleOp module, ConversionPatternRewriter &rewriter,
                    LLVM::LLVMFuncOp mainFunc, llvm::StringRef initialName) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto registerGlobalFunc = getOrCreateRTRegisterGlobals(module, rewriter);
  auto mainFuncName =
      getOrCreateGlobalMainFuncSignature(module, rewriter, initialName);
  rewriter.setInsertionPointToStart(&registerGlobalFunc.getBody().back());
  auto loc = registerGlobalFunc.getLoc();
  // Call runtime to setup argument
  auto mainFuncPtr = rewriter.create<LLVM::AddressOfOp>(loc, mainFunc);
  auto mainFuncNamePtr = rewriter.create<LLVM::AddressOfOp>(loc, mainFuncName);
  auto zero = rewriter.create<LLVM::ZeroOp>(loc, rewriter.getI32Type());
  rewriter.create<LLVM::CallOp>(
      loc, getOrCreateRTFunctionRegister(module, rewriter),
      ArrayRef<Value>{registerGlobalFunc.getArguments()[0], mainFuncPtr,
                      mainFuncNamePtr, mainFuncNamePtr, zero});
}

LLVM::LLVMFuncOp emitStubBody(ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMFuncOp &kernelFunc) {
  LLVM_DEBUG(llvm::dbgs() << "Making stub body\n";);
  OpBuilder::InsertionGuard guard(rewriter);
  // Create the stub body function
  rewriter.setInsertionPointToStart(module.getBody());
  auto llvmVoidTy = rewriter.getType<LLVM::LLVMVoidType>();
  auto bodyStub = rewriter.create<LLVM::LLVMFuncOp>(
      module.getLoc(), kernelFunc.getSymName().str() + "_stub",
      LLVM::LLVMFunctionType::get(llvmVoidTy,
                                  (kernelFunc.getFunctionType().getParams())));

  LLVM_DEBUG(llvm::dbgs() << "\tSetting up stub\n";);
  auto loc = bodyStub.getLoc();
  // Insert a return void terminator
  auto *headBlock = bodyStub.addEntryBlock(rewriter);
  auto *endBlock = rewriter.createBlock(&bodyStub.getBody());
  rewriter.setInsertionPointToEnd(endBlock);
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  rewriter.setInsertionPointToEnd(headBlock);
  const int argWidth = 8;
  auto setupArgumentFunc = getOrCreateRTSetupArgument(module, rewriter);
  int offset = 0;
  Block *currentBlock = headBlock;
  SmallVector<Value> allocatedArgs = storeArgs(rewriter, bodyStub);

  LLVM_DEBUG(llvm::dbgs() << "\tSetting up allocated arguments\n";);
  for (const auto &arg : allocatedArgs) {
    // Set insertion point before the next block
    rewriter.setInsertionPointToEnd(currentBlock);

    // Call runtime to setup argument
    auto callRes = rewriter.create<LLVM::CallOp>(
        loc, setupArgumentFunc,
        ArrayRef<Value>{
            arg,
            rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(
                    argWidth)), // TODO: Change this with proper with
            rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(offset))});

    offset += argWidth;

    assert(callRes.getResults().size() == 1);
    // Compare the result of the call to 0
    auto cmpRes = rewriter.create<LLVM::ICmpOp>(
        loc, rewriter.getI1Type(), LLVM::ICmpPredicate::eq,
        callRes.getResults()[0],
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(0)));
    auto *nextBlock =
        rewriter.createBlock(&bodyStub.getBody(), Region::iterator(endBlock));
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, cmpRes, nextBlock, endBlock);
    currentBlock = nextBlock;
  }

  LLVM_DEBUG(
      llvm::dbgs()
          << "\tArgs allocated, creating final block to launch the kernel\n";);

  // Final block for launching the kernel
  auto glFuncName = getOrCreateGlobalMainFuncName(module, rewriter, kernelFunc);
  rewriter.setInsertionPointToEnd(currentBlock);
  // Call to get or set block number
  auto blockNumSetter = rewriter.create<LLVM::CallOp>(
      loc, getOrCreateGetOrSetBlockNum(module, rewriter),
      ArrayRef<Value>{
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(0)),
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(1))});
  assert(blockNumSetter.getResults().size() == 1);
  // Call to launch the kernel
  rewriter.create<LLVM::CallOp>(
      loc, getOrCreateRTLaunch(module, rewriter),
      ArrayRef<Value>{
          rewriter.create<LLVM::AddressOfOp>(loc, bodyStub),
          rewriter.create<LLVM::AddressOfOp>(loc, glFuncName),
          rewriter.create<LLVM::ConstantOp>(
              loc, rewriter.getI64Type(),
              rewriter.getI64IntegerAttr(bodyStub.getSymName().str().size())),
          blockNumSetter.getResults()[0]});
  rewriter.create<LLVM::BrOp>(loc, endBlock);
  return bodyStub;
}

//===----------------------------------------------------------------------===//
// LaunchOp Lowering
//===----------------------------------------------------------------------===//
// Note: tensor can not lower to llvm, ensure bufferization run before lowering
struct LaunchOpLowering : public ConvertOpToLLVMPattern<LLVM::LLVMFuncOp> {
  using ConvertOpToLLVMPattern<LLVM::LLVMFuncOp>::ConvertOpToLLVMPattern;

  std::string filepath;

  explicit LaunchOpLowering(LLVMTypeConverter &converter, std::string filepath)
      : ConvertOpToLLVMPattern<LLVM::LLVMFuncOp>(converter) {
    this->filepath = std::move(filepath);
  }

  LogicalResult match(LLVM::LLVMFuncOp op) const final {
    return success(hacc::utils::isHost(op) && isCallingDevice(op));
  }

  /// This function will rewrite original host function into a function that is
  /// callable by the user, the main purpose is to setup arguments and to call
  /// the stub function.
  void rewrite(LLVM::LLVMFuncOp originalHostFunc, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto module = originalHostFunc->getParentOfType<ModuleOp>();
    Location loc = originalHostFunc.getLoc();

    // Rename original host function to avoid name conflicts
    const std::string hostFuncName = originalHostFunc.getSymName().str();
    originalHostFunc.setName(hostFuncName + "_old");

    // Set up initialization functions and global variables
    setupInitializationFunctions(module, rewriter);

    // Create new host function with modified signature
    auto newHostFunc = createNewHostFunction(module, rewriter, hostFuncName);

    // Set up function blocks and argument storage
    Block *entryBlock = newHostFunc.addEntryBlock(rewriter);
    Block *exitBlock = rewriter.createBlock(&newHostFunc.getBody());
    setupExitBlock(rewriter, loc, exitBlock);

    // Store function arguments in allocated memory
    SmallVector<Value> storedHostArgs = storeArgs(rewriter, newHostFunc);

    rewriter.setInsertionPointToEnd(entryBlock);
    // Map original arguments to new argument structure
    SmallVector<Value> loadedArguments = setupLoadedNewArguments(
        rewriter, loc, originalHostFunc, storedHostArgs);

    LLVM_DEBUG(llvm::dbgs() << newHostFunc << "\n";);
    // Clone blocks and remap arguments
    cloneBlocksAndRemapArguments(rewriter, loc, originalHostFunc, newHostFunc,
                                 loadedArguments, entryBlock, exitBlock);

    LLVM_DEBUG(llvm::dbgs() << newHostFunc << "\n";);
    // Process kernel calls
    processKernelCalls(module, rewriter, loc, newHostFunc, storedHostArgs,
                       loadedArguments, entryBlock, exitBlock);

    // Replace original function with new implementation
    rewriter.replaceOp(originalHostFunc, newHostFunc);
  }

private:
  /// This register importants functions for kernels inside this host
  /// function
  ///
  /// CCEModuleCtor is the module constructor function that gets called
  /// automatically during program initialization (via llvm.global_ctors).
  /// See @llvm.global_ctors documentation
  ///
  /// https://llvm.org/docs/LangRef.html#id2105
  /// CCEModuleDtor is the destructor
  /// __cce_ptc_wrapper is a structure that wraps the device binary
  /// (PTC - Processor Type Core) information.
  void setupInitializationFunctions(ModuleOp module,
                                    ConversionPatternRewriter &rewriter) const {
    auto registerGlobalFunc = getOrCreateRTRegisterGlobals(module, rewriter);
    auto cceWrapper = getOrCreateCCEPTCWrapper(module, rewriter, filepath);
    auto ctorFunc = getOrCreateCCEModuleCtor(module, rewriter, cceWrapper,
                                             registerGlobalFunc);
    getOrCreateGlobalCtors(module, rewriter, ctorFunc);
  }

  LLVM::LLVMFuncOp createNewHostFunction(ModuleOp module,
                                         ConversionPatternRewriter &rewriter,
                                         StringRef funcName) const {
    auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
    auto llvmVoidTy = rewriter.getType<LLVM::LLVMVoidType>();

    // New host function signature: [BlockNum, l2def, stream, Args]
    return getOrCreateFuncOp(
        module, rewriter, funcName, llvmVoidTy,
        {rewriter.getI32Type(), llvmPtrTy, llvmPtrTy, llvmPtrTy});
  }

  void setupExitBlock(ConversionPatternRewriter &rewriter, Location loc,
                      Block *exitBlock) const {
    rewriter.setInsertionPointToEnd(exitBlock);
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  SmallVector<Value>
  setupLoadedNewArguments(ConversionPatternRewriter &rewriter, Location loc,
                          LLVM::LLVMFuncOp originalHostFunc,
                          SmallVector<Value> &storedHostArgs) const {
    SmallVector<Value> loadedArguments;
    auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();
    auto one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                 rewriter.getI32IntegerAttr(1));

    // Access the tensor arguments array (4th parameter of new host function)
    // storedHostArgs[3] contains pointer to array of all tensor arguments
    // loadFromPtr dereferences this pointer to get the actual array
    Value tensorArgs = loadFromPtr(loc, rewriter, storedHostArgs[3]);

    // Iterate through all arguments of the original host function
    // gepIndex is used for GEP array indexing, arg is the current argument
    for (auto [gepIndex, arg] :
         llvm::enumerate(originalHostFunc.getArguments())) {
      // Creates a stack allocation (alloca) for the current argument
      // llvmPtrTy: pointer type for the allocation
      // arg.getType(): type of the current argument
      // one: constant value 1 for single element allocation
      Value allocated =
          rewriter.create<LLVM::AllocaOp>(loc, llvmPtrTy, arg.getType(), one);

      // Calculate pointer to the current argument in the tensor args
      // array GEP (GetElementPtr) calculates address of array element at
      // gepIndex Parameters:
      // - loc: source location for debugging
      // - llvmPtrTy: result type (pointer)
      // - llvmPtrTy: source pointer type
      // - tensorArgs: base pointer to array
      // - gepIndex: index into array (cast to i64)
      // - true: inbounds flag for optimization
      LLVM::GEPOp gep = rewriter.create<LLVM::GEPOp>(
          loc, llvmPtrTy, llvmPtrTy, tensorArgs,
          ArrayRef<LLVM::GEPArg>{static_cast<int64_t>(gepIndex)}, true);

      // Load the argument value and store it in local allocation
      // First load the actual argument value from the calculated GEP address
      Value argValue = loadFromPtr(loc, rewriter, gep);
      // Store the loaded value into our local allocation
      rewriter.create<LLVM::StoreOp>(loc, argValue, allocated);
      Value loaded = loadFromPtr(loc, rewriter, allocated);

      // Mark original argument to its local allocation
      loadedArguments.push_back(loaded);
    }
    return loadedArguments;
  }

  void handleCondBrOp(LLVM::CondBrOp condBrOp,
                      ConversionPatternRewriter &rewriter,
                      IRMapping &valueMapping,
                      DenseMap<Block *, Block *> &blockMapping) const {
    Value mappedCondition =
        valueMapping.lookupOrDefault(condBrOp.getCondition());

    // Get mapped destination blocks
    Block *newTrueDest = blockMapping[condBrOp.getTrueDest()];
    Block *newFalseDest = blockMapping[condBrOp.getFalseDest()];
    SmallVector<Value> newTrueOperands;
    SmallVector<Value> newFalseOperands;

    for (Value operand : condBrOp.getTrueDestOperands()) {
      newTrueOperands.push_back(valueMapping.lookupOrDefault(operand));
    }
    for (Value operand : condBrOp.getFalseDestOperands()) {
      newFalseOperands.push_back(valueMapping.lookupOrDefault(operand));
    }

    // Preserve branch weights if present
    DenseI32ArrayAttr branchWeights = condBrOp.getBranchWeightsAttr();

    rewriter.create<LLVM::CondBrOp>(condBrOp->getLoc(), mappedCondition,
                                    newTrueOperands, newFalseOperands,
                                    branchWeights, newTrueDest, newFalseDest);
  };

  void handleBrOp(LLVM::BrOp brOp, ConversionPatternRewriter &rewriter,
                  IRMapping &valueMapping,
                  DenseMap<Block *, Block *> &blockMapping) const {
    Block *newDest = blockMapping[brOp.getDest()];
    SmallVector<Value> newOperands;
    for (Value operand : brOp.getDestOperands()) {
      newOperands.push_back(valueMapping.lookupOrDefault(operand));
    }
    rewriter.create<LLVM::BrOp>(brOp->getLoc(), newOperands, newDest);
  };

  void handleReturnOp(LLVM::ReturnOp returnOp,
                      ConversionPatternRewriter &rewriter,
                      IRMapping &valueMapping, Block *exitBlock) const {
    SmallVector<Value> newReturnValues;
    for (Value returnVal : returnOp.getOperands()) {
      newReturnValues.push_back(valueMapping.lookupOrDefault(returnVal));
    }
    rewriter.create<LLVM::BrOp>(returnOp->getLoc(), newReturnValues, exitBlock);
  };

  void handleSwitchOp(LLVM::SwitchOp switchOp,
                      ConversionPatternRewriter &rewriter,
                      IRMapping &valueMapping,
                      DenseMap<Block *, Block *> &blockMapping) const {
    SmallVector<Value> newOperands;

    // Condition
    newOperands.push_back(valueMapping.lookupOrDefault(switchOp.getValue()));

    // Mapped default operands
    for (Value operand : switchOp.getDefaultOperands()) {
      newOperands.push_back(valueMapping.lookupOrDefault(operand));
    }

    // Mapped case operands
    for (auto caseOperands : switchOp.getCaseOperands()) {
      for (Value operand : caseOperands) {
        newOperands.push_back(valueMapping.lookupOrDefault(operand));
      }
    }

    auto newSwitch = rewriter.create<LLVM::SwitchOp>(
        switchOp->getLoc(), newOperands, switchOp.getSuccessors(),
        switchOp->getAttrs() // Attributes do not need remapping
    );

    // Remap block successors
    for (unsigned i = 0; i < switchOp.getNumSuccessors(); ++i) {
      newSwitch->setSuccessor(blockMapping[switchOp.getSuccessor(i)], i);
    };
  };

  void handleDefault(Operation *defaultOp, ConversionPatternRewriter &rewriter,
                     IRMapping &valueMapping) const {
    Operation *clonedOp = rewriter.clone(*defaultOp, valueMapping);
    for (auto [origResult, clonedResult] :
         llvm::zip(defaultOp->getResults(), clonedOp->getResults())) {
      valueMapping.map(origResult, clonedResult);
    }
  };

  void cloneBlocksAndRemapArguments(ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    LLVM::LLVMFuncOp originalHostFunc,
                                    LLVM::LLVMFuncOp newHostFunc,
                                    SmallVector<Value> &loadedArguments,
                                    Block *entryBlock, Block *exitBlock) const {
    IRMapping valueMapping;
    DenseMap<Block *, Block *> blockMapping;

    // Map original arguments to loaded arguments
    for (auto [origArg, loadedArg] :
         llvm::zip(originalHostFunc.getArguments(), loadedArguments)) {
      valueMapping.map(origArg, loadedArg);
    }

    // Create blocks and map arguments
    for (Block &origBlock : originalHostFunc.getBlocks()) {
      Block *newBlock;
      if (&origBlock == &originalHostFunc.front()) {
        // For entry block, clone operations into existing entry block
        newBlock = entryBlock;
      } else {
        // Create new block before exit block
        newBlock = rewriter.createBlock(entryBlock->getParent(),
                                        Region::iterator(exitBlock),
                                        origBlock.getArgumentTypes());

        // Map block arguments
        for (auto [origArg, newArg] :
             llvm::zip(origBlock.getArguments(), newBlock->getArguments())) {
          valueMapping.map(origArg, newArg);
        }
      }
      blockMapping[&origBlock] = newBlock;
    }

    // Clone operations and handle control flow
    for (Block &origBlock : originalHostFunc.getBlocks()) {
      Block *newBlock = blockMapping[&origBlock];
      rewriter.setInsertionPointToEnd(newBlock);
      for (Operation &op : origBlock) {
        TypeSwitch<Operation *>(&op)
            .Case<LLVM::CondBrOp>([&](auto condBrOp) {
              handleCondBrOp(condBrOp, rewriter, valueMapping, blockMapping);
            })
            .Case<LLVM::BrOp>([&](auto brOp) {
              handleBrOp(brOp, rewriter, valueMapping, blockMapping);
            })
            .Case<LLVM::ReturnOp>([&](auto returnOp) {
              handleReturnOp(returnOp, rewriter, valueMapping, exitBlock);
            })
            .Case<LLVM::SwitchOp>([&](auto switchOp) {
              handleSwitchOp(switchOp, rewriter, valueMapping, blockMapping);
            })
            .Default([&](Operation *defaultOp) {
              handleDefault(defaultOp, rewriter, valueMapping);
            });
      }
    }
  }

  void processKernelCalls(ModuleOp module, ConversionPatternRewriter &rewriter,
                          Location loc, LLVM::LLVMFuncOp newHostFunc,
                          SmallVector<Value> &storedHostArgs,
                          SmallVector<Value> &loadedArguments,
                          Block *entryBlock, Block *exitBlock) const {
    // Get or create the kernel configuration setup function
    auto kernelConfigFunc = getOrCreatePrepareRTConfigureCall(module, rewriter);
    Block *currentBlock = entryBlock;

    // Collect all kernel calls first to avoid recursive modification
    SmallVector<LLVM::CallOp> kernelCalls;
    newHostFunc.walk(
        [&](LLVM::CallOp callOp) { kernelCalls.push_back(callOp); });

    // Process each collected kernel call
    for (auto kernelCall : kernelCalls) {
      rewriter.setInsertionPoint(kernelCall);

      auto calleeOp =
          mlir::utils::getCalledFunction<hacc::HACCFunction, LLVM::CallOp>(
              kernelCall);
      if (calleeOp.isDevice()) {
        LDBG("Changing " << calleeOp << "\n");
        // Set up kernel configuration
        auto configBlock = setupKernelConfig(module, rewriter, loc, kernelCall,
                                             storedHostArgs, kernelConfigFunc);

        // Create kernel stub and register it
        (void)createAndRegisterKernelStub(module, rewriter, loc, kernelCall);

        // Link blocks together
        linkBlocks(rewriter, loc, currentBlock, configBlock.configBlock,
                   configBlock.execBlock, exitBlock);
        rewriter.eraseOp(kernelCall);
        currentBlock = configBlock.execBlock;
      }

      LLVM_DEBUG(llvm::dbgs() << newHostFunc << "\n";);
      // Connect final block to exit
      if (!currentBlock->mightHaveTerminator()) {
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<LLVM::BrOp>(loc, exitBlock);
      }
    }
  }

  struct ConfigBlockInfo {
    Block *configBlock; // Block containing configuration code
    Block *execBlock;   // Block containing kernel execution code
  };

  ConfigBlockInfo setupKernelConfig(ModuleOp module,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc, LLVM::CallOp kernelCall,
                                    SmallVector<Value> &storedHostArgs,
                                    LLVM::LLVMFuncOp kernelConfigFunc) const {
    rewriter.setInsertionPoint(kernelCall);

    // Load configuration parameters
    SmallVector<Value> configParams =
        loadConfigParameters(rewriter, loc, storedHostArgs);

    // Handle static block dimensions if present
    auto kernelFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        kernelCall.getCalleeAttr().getValue());
    configParams[0] =
        getBlockDimension(rewriter, loc, kernelFunc, configParams[0]);

    // Create configuration call
    auto configCall =
        rewriter.create<LLVM::CallOp>(loc, kernelConfigFunc, configParams);

    // Create comparison for configuration success
    createConfigCheck(rewriter, loc, configCall.getResult());

    // Split blocks for configuration and execution
    auto *topBlock = rewriter.getInsertionBlock();
    auto *bottomBlock =
        rewriter.splitBlock(topBlock, rewriter.getInsertionPoint());

    return {topBlock, bottomBlock};
  }

  SmallVector<Value>
  loadConfigParameters(ConversionPatternRewriter &rewriter, Location loc,
                       SmallVector<Value> &storedHostArgs) const {
    SmallVector<Value> configParams;
    for (int i = 0; i < 3; i++) { // config parameter of kernel call has 3 args
      configParams.push_back(loadFromPtr(loc, rewriter, storedHostArgs[i]));
    }
    return configParams;
  }

  Value getBlockDimension(ConversionPatternRewriter &rewriter, Location loc,
                          LLVM::LLVMFuncOp kernelFunc, Value defaultDim) const {
    if (auto blockDimAttr = kernelFunc->getAttr(hacc::BlockDimAttr::name)) {
      if (auto blockDimIntAttr = dyn_cast<mlir::IntegerAttr>(blockDimAttr)) {
        return rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(blockDimIntAttr.getInt()));
      }
    }
    return defaultDim;
  }

  Value createConfigCheck(ConversionPatternRewriter &rewriter, Location loc,
                          Value configResult) const {
    return rewriter.create<LLVM::ICmpOp>(
        loc, rewriter.getI1Type(), LLVM::ICmpPredicate::eq, configResult,
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(0)));
  }

  LLVM::LLVMFuncOp
  createAndRegisterKernelStub(ModuleOp module,
                              ConversionPatternRewriter &rewriter, Location loc,
                              LLVM::CallOp kernelCall) const {
    auto kernelFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(
        kernelCall.getCalleeAttr().getValue());

    // Create kernel stub
    auto kernelStub = emitStubBody(module, rewriter, kernelFunc);
    // Prepare operands for the kernel call
    SmallVector<Value> kernelOperands = kernelCall->getOperands();

    // Create the kernel call
    rewriter.setInsertionPoint(kernelCall);
    rewriter.create<LLVM::CallOp>(loc, kernelStub, kernelOperands);

    // Register the kernel
    registerKernel(module, rewriter, kernelStub, kernelFunc.getSymName());

    LLVM_DEBUG(llvm::dbgs() << kernelStub << "\n";);
    return kernelStub;
  }

  void linkBlocks(ConversionPatternRewriter &rewriter, Location loc,
                  Block *currentBlock, Block *configBlock, Block *execBlock,
                  Block *exitBlock) const {
    auto *configCheck = &configBlock->back();
    LLVM_DEBUG(llvm::dbgs() << *configCheck << "\n";);

    if (!currentBlock->mightHaveTerminator()) { // No Tiled Kernel
      rewriter.setInsertionPointToEnd(currentBlock);
    } else {
      loc = configCheck->getLoc();
      rewriter.setInsertionPointToEnd(configBlock);
    }

    auto cmpOp = cast<LLVM::ICmpOp>(configCheck);
    rewriter.create<LLVM::CondBrOp>(loc, cmpOp.getResult(), execBlock,
                                    exitBlock);
  }
};

void mlir::hacc::populateHACCToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const std::string &deviceFilePath) {
  patterns.add<LaunchOpLowering>(converter, deviceFilePath);
}

std::unique_ptr<Pass> mlir::createConvertHACCToLLVMPass() {
  return std::make_unique<ConvertHACCToLLVM>();
}

std::unique_ptr<Pass>
mlir::createConvertHACCToLLVMPass(const ConvertHACCToLLVMOptions &option) {
  return std::make_unique<ConvertHACCToLLVM>(option);
}
