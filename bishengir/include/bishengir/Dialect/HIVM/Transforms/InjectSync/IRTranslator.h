//===------------- IRTranslator.h ----Sync information collection ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_IRTRANSLATOR_H
#define BISHENGIR_IRTRANSLATOR_H

#include "bishengir/Dialect/HIVM/Transforms/InjectSync/MemoryDependentAnalyzer.h"
#include "bishengir/Dialect/HIVM/Transforms/InjectSync/SyncCommon.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace hivm {

class IRTranslator {
public:
  IRTranslator(SyncIRs &syncIR, MemoryDependentAnalyzer &memDepAnalyzer,
               Buffer2MemInfoMap &buffer2MemInfoMap, func::FuncOp func,
               SyncAnalysisMode syncAnalysisMode)
      : syncIR(syncIR), memAnalyzer(memDepAnalyzer),
        syncAnalysisMode(syncAnalysisMode),
        buffer2MemInfoMap(std::move(buffer2MemInfoMap)), func_(func){};

  virtual ~IRTranslator() = default;

  /// Build entrance.
  void Build();

  /// Recursive traversal to collect IR information.
  virtual void RecursionIR(Region *region);

  /// Get buffer2ParentAliasBuffer.
  DenseMap<Value, Value> GetBuffer2ParentAliasBuffer() {
    return buffer2ParentAliasBuffer;
  }

public:
  /// Update global BaseMemInfo alias info.
  void UpdateKernelArgMemInfo();

  /// Collect information on ForOp, handle for begin inform.
  void UpdateForOpInfo(scf::ForOp forOp);

  /// Collect information on ForOp, handle for begin inform.
  void UpdateWhileOpInfo(scf::WhileOp whileOp);

  /// Collect information on IfOp, handle if begin inform.
  void UpdateIfOpInform(scf::IfOp ifOp);

  /// Update BaseMemInfo for defVec and useVec.
  void UpdateDefUseVec(const SmallVector<Value> &inOutVals,
                       SmallVector<const BaseMemInfo *> &memInfoVec);

  /// update the result buffer mem info of alloc like op
  LogicalResult UpdateAllocLikeOpMemInfo(Operation *op);

  /// Update forOp InitArgs and RegionIterArgs alias info.
  void UpdateForInitArgsAliasInfo(scf::ForOp forOp);

  /// Update whileOp InitArgs and RegionIterArgs alias info.
  void UpdateWhileInitArgsAliasInfo(scf::WhileOp whileOp);

  /// Update whileOp Result and RegionIterArgs alias info.
  void UpdateWhileResultAliasInfo(scf::WhileOp whileOp);

  /// Collect information on result replace source baseAddress and allocate
  /// size.
  void
  UpdateAliasBufferInfo(Value result, Value source,
                        std::optional<std::reference_wrapper<Buffer2MemInfoMap>>
                            buffer2MemInfoMapOpt = {});

  void UpdateCallOp(func::CallOp callOp);

  /// Determine whether the loadOp is from tensor extract op.
  bool isTensorExtractLoadOp(Operation *op);

  /// Save the Global syncIR.
  SyncIRs &syncIR;

  /// Save the baseMemInfo entity and determines memory conflicts.
  MemoryDependentAnalyzer &memAnalyzer;

  SyncAnalysisMode syncAnalysisMode;

  /// Maps buffers to their mem-info.
  Buffer2MemInfoMap buffer2MemInfoMap;

  /// Same as buffer2MemInfoMap but includes work-space arguments.
  Buffer2MemInfoMap buffer2MemInfoMapIncludingWSArgs;

  /// A reference to the function being processed.
  func::FuncOp func_;

  /// The serial index of syncIR.
  uint32_t index{0};

  /// Record the relationship between buffer and alias by buffer.
  DenseMap<Value, Value> buffer2ParentAliasBuffer;

private:
  /// Collect information on constantOp, like: %c0_i64=arith.constant 0:i64
  void UpdateConstantOpInform(arith::ConstantOp constOp);

  /// Collect information on YieldOp, handle if yield and for yield.
  void UpdateYieldOpInform(scf::YieldOp yieldOp);

  /// Collect information on DestinationStyleOpInterface, handle instruction
  /// inform.
  void
  UpdateDestinationStyleOpInterfaceInform(Operation *op,
                                          DestinationStyleOpInterface dstOp);

  /// Collect information on load or store op.
  template <typename OP>
  typename std::enable_if<std::is_same_v<OP, memref::LoadOp> ||
                              std::is_same_v<OP, affine::AffineLoadOp> ||
                              std::is_same_v<OP, affine::AffineStoreOp> ||
                              std::is_same_v<OP, memref::StoreOp>,
                          void>::type
  UpdateStoreOrLoadOpInform(OP op);

  /// Check whether there is an unknown operation with buffer
  /// information.
  LogicalResult CheckIfUnknownOpTouchBuffer(Operation *op) const;

  /// Determine whether the current operation can be skipped.
  bool isSkippableOp(Operation *op) const;

  /// Update temp buffer to defVec.
  void UpdateTempOpDefVec(Operation *op,
                          SmallVector<const BaseMemInfo *> &defVec);

  /// Update BaseMemInfo for defVec.
  void UpdateOpDefVec(DestinationStyleOpInterface dstOp,
                      SmallVector<const BaseMemInfo *> &defVec);

  /// Update BaseMemInfo for useVec.
  void UpdateOpUseVec(DestinationStyleOpInterface dstOp,
                      SmallVector<const BaseMemInfo *> &useVec);

  void UpdateOpDefUseVec(func::CallOp callOp,
                         SmallVector<const BaseMemInfo *> &defVec,
                         SmallVector<const BaseMemInfo *> &useVec);

  void UpdateFuncArguments(func::FuncOp funcOp,
                           Buffer2MemInfoMap &buffer2MemInfoMap);

  /// Update the src and dst information of MacroOp.
  void UpdateMacroOpInform(DestinationStyleOpInterface dstOp);

  void UpdateGPULaunchFuncOpInform(gpu::LaunchFuncOp);

  void UpdateOpDefUseVec(gpu::LaunchFuncOp gpuLaunchFunc,
                         SmallVector<const BaseMemInfo *> &defVec,
                         SmallVector<const BaseMemInfo *> &useVec);

  /// Insert a place-holder instance-element.
  void InsertPlaceHolderInst(InstanceElement *parentScope);

private:
  /// The actual base address corresponding to the buffer.
  /// note: multiBuffer has multiple addresses.
  DenseMap<Value, SmallVector<uint64_t>> buffer2BaseAddresses;

  /// The actual allocate size corresponding to the buffer.
  DenseMap<Value, uint64_t> buffer2AllocateSize;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_IRTRANSLATOR_H
