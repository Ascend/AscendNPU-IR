//===------------- InjectSync.h ----Auto Inject Sync ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/IRTranslator.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <optional>
namespace mlir {
namespace hivm {

const int64_t blockAllFlagId1 = 8;
const int64_t blockAllFlagId2 = 9;

class InjectBlockSyncAnalysis {
public:
  InjectBlockSyncAnalysis(func::FuncOp func) : func_(func) {}

  /// Inject Shallow block sync.
  void InjectBlockShallowSync();

  /// Inject MixCV block sync.
  void InjectBlockMixSync(bool assumeAliveLoops);

  /// Inject all block sync.
  void InjectAllBlockSync();

private:
  /// Inferring the core type of funcs for op core type.
  TCoreType convertFuncCoreTypeToCoreType(TFuncCoreType funcCoreType);

  /// Inferring op core type.
  std::optional<::mlir::hivm::TCoreType> queryCoreType(Operation *op);

  /// Generate block sync event id.
  IntegerAttr generateFlagId(OpBuilder opBuilder);

  /// Generate block all sync op.
  SyncBlockOp generateSyncBlockOp(OpBuilder opBuilder, Location loc,
                                  IntegerAttr flagId, TCoreType coreType);

  /// Generate block set or wait sync op.
  template <typename OpType>
  OpType generateCVSyncOp(OpBuilder opBuilder, Location loc, TCoreType coreType,
                          PIPE pipe, IntegerAttr flagIdAttr);

  /// Inject block sync between op.
  void injectSyncBetweenOp(OpBuilder &opBuilder, Operation *op,
                           TCoreType opCoreType,
                           SetVector<TCoreType> &userOpCoreTypes);

  /// Inject block sync op.
  LogicalResult injectShallowBlockSync(Operation *op);

private:
  func::FuncOp func_;

  /// Block sync event id.
  uint64_t flagIdCnt{0};
};

class SyncBlockIRTranslator : public IRTranslator {
public:
  SyncBlockIRTranslator(SyncIRs &syncIR,
                        MemoryDependentAnalyzer &memDepAnalyzer,
                        Buffer2MemInfoMap &buffer2MemInfoMap, func::FuncOp func,
                        SyncAnalysisMode syncAnalysisMode)
      : IRTranslator(syncIR, memDepAnalyzer, buffer2MemInfoMap, func,
                     syncAnalysisMode) {};

  ~SyncBlockIRTranslator() = default;

  /// Build entrance.
  void SyncBlockBuild();

  /// Recursive traversal to collect IR information.
  void RecursionIR(Region *region) override;

private:
  /// Collect information on YieldOp, handle if yield and for yield.
  void UpdateYieldOpInform(scf::YieldOp yieldOp);

  /// Update the tensor dst and result alias.
  void UpdateInitAndResAlias(DestinationStyleOpInterface dstStyleOp);

  /// Collect information on DestinationStyleOpInterface, handle instruction
  /// inform.
  void UpdateDestinationStyleOpInform(Operation *op,
                                      DestinationStyleOpInterface dstStyleOp);

  /// Collect information on tensor.extract op.
  void UpdateTensorExtractOpInform(Operation *op, tensor::ExtractOp extractOp);

  /// Collect information on load or store op.
  template <typename OP> void UpdateStoreOrLoadOpInfoBlockSync(OP op);

  /// Collect information on alloc ops accessed cross-core.
  void UpdateAllocOpMeminfo(memref::AllocOp allocOp);

  bool isVectorOpResult(Value value);

  std::optional<hivm::PIPE>
  getInferredPipe(Operation *op, TCoreType coreType,
                  const SmallVector<const BaseMemInfo *> &defVec);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H