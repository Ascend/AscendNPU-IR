//===------------- InjectSync.h ----Auto Inject Sync ----------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H

#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/IRTranslator.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MoveSyncState.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/RemoveRedundantSync.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncAnalysis.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCodegen.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncDebug.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncEventIdAllocation.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
namespace hivm {

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
                     syncAnalysisMode){};

  ~SyncBlockIRTranslator() = default;

  /// Build entrance.
  void SyncBlockBuild();

  /// Recursive traversal to collect IR information.
  void RecursionIR(Region *region) override;

private:
  /// Collect information on YieldOp, handle if yield and for yield.
  void UpdateYieldOpInform(scf::YieldOp yieldOp);

  /// Update the buffer information of alias.
  void UpdateAliasBufferInfo(Value buffer, Value aliasBuffer) override;

  /// Update the tensor dst and result alias.
  void UpdateInitAndResAlias(DestinationStyleOpInterface dstStyleOp);

  /// Collect information on DestinationStyleOpInterface, handle instruction
  /// inform.
  void UpdateDestinationStyleOpInform(Operation *op,
                                      DestinationStyleOpInterface dstStyleOp);

  void UpdateTensorExtractOpInform(Operation *op, tensor::ExtractOp extractOp);
};

} // namespace hivm
} // namespace mlir

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_INJECT_BLOCK_SYNC_H