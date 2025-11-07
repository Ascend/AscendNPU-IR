//===- BiShengSegmenter.cpp - Pass to generate some segmented funcOp ------===//
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
#include "Test/TestPasses.h"

#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"

#include <set>

#define DEBUG_TYPE "bisheng-segmenter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
namespace bishengir_test {
using namespace mlir;

/// A pass that segments functions into smaller sub-functions.
/// Each sub-function contains a window of consecutive operations from the
/// original function.
struct BiShengSegmenterPass
    : public PassWrapper<BiShengSegmenterPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BiShengSegmenterPass)

  BiShengSegmenterPass() = default;
  BiShengSegmenterPass(const BiShengSegmenterPass &other) {
    segmentSize = other.segmentSize;
    preserveOriginal = other.preserveOriginal;
    segmentPrefix = other.segmentPrefix;
  }

  BiShengSegmenterPass &operator=(const BiShengSegmenterPass &other) {
    if (this != &other) {
      *this = BiShengSegmenterPass(other); // Copy-and-swap idiom
    }
    return *this;
  }

  /// Pass options structure
  Option<int> segmentSize{
      *this, "segment-size",
      llvm::cl::desc("Specify the number of operations per segment"),
      llvm::cl::init(1)};

  // Additional options can be added here
  Option<bool> preserveOriginal{
      *this, "preserve-original",
      llvm::cl::desc("Keep the original function alongside segments"),
      llvm::cl::init(false)};

  Option<std::string> segmentPrefix{
      *this, "segment-prefix",
      llvm::cl::desc("Prefix for generated segment function names"),
      llvm::cl::init("seg")};

  StringRef getArgument() const final { return DEBUG_TYPE; }
  StringRef getDescription() const final {
    return "Segments functions into smaller sub-functions";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functionsToProcess;
    collectEligibleFunctions(module, functionsToProcess);

    for (auto funcOp : functionsToProcess) {
      if (failed(processFunction(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

  /// Collects functions that can be segmented (single-block functions).
  void collectEligibleFunctions(ModuleOp module,
                                SmallVector<func::FuncOp> &functions) {
    module.walk([&](func::FuncOp op) {
      if (op.getBlocks().size() == 1) {
        functions.push_back(op);
      } else {
        LDBG("Skipping multi-block function: " << op.getName());
      }
    });
  }

  /// Processes a single function, creating segmented versions.
  LogicalResult processFunction(func::FuncOp funcOp) {
    if (failed(optimizeFunction(funcOp))) {
      return failure();
    }
    Block &block = funcOp.getBlocks().front();
    sortTopologically(&block);

    // Collect all operations (excluding terminator)
    SmallVector<Operation *> operations;
    collectNonTerminatorOps(block, operations);

    int numOps = static_cast<int>(operations.size());
    if (segmentSize == 0) {
      LDBG("Invalid segment size: " << segmentSize);
      return failure();
    }
    for (int startIdx = 0; startIdx + segmentSize - 1 < numOps; startIdx++) {
      createSegmentFunction(funcOp, startIdx, startIdx + segmentSize - 1);
    }

    // Remove the original function if not preserving
    if (!preserveOriginal) {
      funcOp->erase();
    }
    return success();
  }

  LogicalResult optimizeFunction(func::FuncOp funcOp) {
    PassManager pm(funcOp.getContext());
    pm.addPass(createCSEPass());
    CanonicalizerOptions options;
    SmallVector<std::string> disabledPatterns(
        {"ReinterpretCastConstantArgumentFolder"});
    options.disabledPatterns = disabledPatterns;
    pm.addPass(bishengir::createExtendedCanonicalizerPass(options));
    return pm.run(funcOp);
  }

  /// Collects all non-terminator operations from a block.
  void collectNonTerminatorOps(Block &block,
                               SmallVector<Operation *> &operations) {
    for (Operation &op : block.getOperations()) {
      if (!op.hasTrait<OpTrait::IsTerminator>()) {
        operations.push_back(&op);
      }
    }
  }

  /// Creates a single segment function containing operations [startIdx,
  /// endIdx].
  void createSegmentFunction(func::FuncOp originalFunc, int startIdx,
                             int endIdx) {
    OpBuilder builder(&getContext());
    builder.setInsertionPoint(originalFunc);

    // Clone the original function
    func::FuncOp segmentFunc = cast<func::FuncOp>(builder.clone(*originalFunc));
    Block &entryBlock = segmentFunc.getBlocks().front();
    SmallVector<Operation *> clonedOps;
    collectNonTerminatorOps(entryBlock, clonedOps);
    isolateOperations(segmentFunc, clonedOps, startIdx, endIdx);
    ArrayRef<Operation *> slicedOps = clonedOps;
    slicedOps = slicedOps.slice(startIdx, endIdx - startIdx + 1);
    // Create return values and update function signature
    updateSegmentSignature(segmentFunc, slicedOps);
    std::string segmentName =
        generateSegmentName(originalFunc.getName(), startIdx, segmentFunc);
    segmentFunc.setName(segmentName);
    LDBG(segmentFunc);
  }

  void isolateOperations(func::FuncOp segmentFunc,
                         ArrayRef<Operation *> operations, int startIdx,
                         int endIdx) {
    // Erasing terminator
    Block &entryBlock = segmentFunc.getBlocks().front();
    entryBlock.getTerminator()->erase();
    LDBG("Starting to isolate from " << startIdx << " " << endIdx);
    LDBG("Peek front " << *operations.front());
    LDBG("Peek back " << *operations.back());
    for (int i = static_cast<int>(operations.size()) - 1; i > endIdx; --i)
      operations[i]->erase();
    LDBG("Pruned back");
    for (int i = startIdx - 1; i >= 0; --i) {
      Operation *op = operations[i];
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          Value newArg =
              entryBlock.addArgument(result.getType(), segmentFunc->getLoc());
          result.replaceAllUsesWith(newArg);
        }
      }
      op->erase();
    }
    LDBG("Pruned front");
    int argSize = static_cast<int>(entryBlock.getNumArguments());
    for (int i = argSize - 1; i >= 0; i--) {
      if (entryBlock.getArgument(i).use_empty()) {
        entryBlock.eraseArgument(i);
      }
    }
    LDBG("Pruned args");
  }

  /// Updates the function signature with new arguments and return values.
  void updateSegmentSignature(func::FuncOp segmentFunc,
                              ArrayRef<Operation *> operations) {
    LDBG("Construct new return");
    SmallVector<Value> returnValues;
    for (size_t i = 0; i < operations.size(); ++i) {
      for (Value result : operations[i]->getResults()) {
        if (result.use_empty()) {
          returnValues.push_back(result);
        }
      }
    }

    OpBuilder builder(&getContext());
    Block &entryBlock = segmentFunc.getBlocks().front();
    builder.setInsertionPointToEnd(&entryBlock);
    builder.create<func::ReturnOp>(segmentFunc.getLoc(), returnValues);
    LDBG("ReturnOp made");
    auto returnTypes = llvm::to_vector(
        llvm::map_range(returnValues, [](Value v) { return v.getType(); }));
    auto argTypes = entryBlock.getArgumentTypes();
    segmentFunc.setFunctionType(builder.getFunctionType(argTypes, returnTypes));

    // Clear argument attributes for new arguments
    segmentFunc.setAllArgAttrs(ArrayAttr::get(
        segmentFunc.getContext(),
        SmallVector<Attribute>(argTypes.size(),
                               DictionaryAttr::get(segmentFunc.getContext()))));
  }

  std::string generateSegmentName(StringRef baseName, int segmentIndex,
                                  func::FuncOp segmentFunc) {
    std::string prefix = segmentPrefix.getValue();
    return (baseName + "_" + prefix + std::to_string(segmentIndex) + "_" +
            std::to_string(hash_value(segmentFunc->hashProperties())))
        .str();
  }
};

void registerBiShengSegmenterPass() {
  PassRegistration<BiShengSegmenterPass>();
}

} // namespace bishengir_test