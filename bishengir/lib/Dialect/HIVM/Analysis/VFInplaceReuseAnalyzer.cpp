//===- VFInplaceReuseAnalyzer.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Analysis/VFInplaceReuseAnalyzer.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::hivm;

#define DEBUG_TYPE "vf-inplace-reuse"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hivm {

/// Creates a new vf-inplace-reuse analysis that computes argument inplace
/// reuse information for all associated vf functions and vf calls.
VFInplaceReuseAnalysis::VFInplaceReuseAnalysis(ModuleOp moduleOp)
    : moduleOp(moduleOp) {
  build();
}

/// Initializes the internal mappings.
void VFInplaceReuseAnalysis::build() {
  SmallVector<func::FuncOp> callerFuncs;
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (isVF(funcOp)) {
      buildVFCalleeFunc(funcOp);
      continue;
    }
    callerFuncs.push_back(funcOp);
  }
  for (auto funcOp : callerFuncs) {
    buildVFCallerFunc(funcOp);
  }
  LLVM_DEBUG(dump());
}

using ReadListT = SetVector<vector::TransferReadOp>;
using WriteListT = SetVector<vector::TransferWriteOp>;
using ReadMapT = DenseMap<BlockArgument, ReadListT>;
using WriteMapT = DenseMap<BlockArgument, WriteListT>;
using TraceListT = SmallVector<Operation *>;

std::optional<Value> getLoopInitArg(scf::ForOp forOp, BlockArgument iterArg) {
  for (const auto &[regionArg, initArg] :
       zip_equal(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
    if (regionArg == iterArg) {
      return initArg;
    }
  }
  return std::nullopt;
}

std::optional<BlockArgument> traceToFuncArg(Value value);

// trace through block argument across forOp iter_args
std::optional<BlockArgument> traceBlockArgument(BlockArgument ba) {
  auto forOp = dyn_cast<scf::ForOp>(ba.getParentBlock()->getParentOp());
  if (!forOp) {
    return ba;
  }
  auto initArg = getLoopInitArg(forOp, ba);
  if (!initArg.has_value()) {
    return std::nullopt;
  }
  return traceToFuncArg(initArg.value());
}

/// trace value to func argument.
std::optional<BlockArgument> traceToFuncArg(Value value) {
  if (auto ba = dyn_cast<BlockArgument>(value)) {
    return traceBlockArgument(ba);
  }
  // support tracing subview and extract_slice at the same time
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    return traceToFuncArg(subview.getViewSource());
  }
  if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    return traceToFuncArg(slice.getSource());
  }
  // TODO: is there any other pattern for load/store
  LDBG("tracing unsupported op: " << value);
  return std::nullopt;
}

void traceFuncArgReadsAndWrites(func::FuncOp funcOp, ReadMapT &arg2reads,
                                WriteMapT &arg2writes) {
  funcOp.walk([&](Operation *op) {
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      auto arg = traceToFuncArg(read.getSource());
      if (!arg.has_value()) {
        return WalkResult::advance();
      }
      arg2reads[arg.value()].insert(read);
    } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      auto arg = traceToFuncArg(write.getSource());
      if (!arg.has_value()) {
        return WalkResult::advance();
      }
      arg2writes[arg.value()].insert(write);
    }
    return WalkResult::advance();
  });
}

SmallVector<vector::TransferReadOp>
getPrevReadsInSameBlock(vector::TransferWriteOp write) {
  SmallVector<vector::TransferReadOp> result;
  auto startIt = write->getBlock()->begin();
  auto endIt = write->getIterator();
  for (Operation &op : llvm::make_range(startIt, endIt)) {
    if (auto read = dyn_cast<vector::TransferReadOp>(&op)) {
      result.push_back(read);
    }
  }
  return result;
}

std::optional<BlockArgument> getReadArg(vector::TransferReadOp target,
                                        ReadMapT arg2reads) {
  for (const auto &[ba, reads] : arg2reads) {
    if (reads.contains(target)) {
      return ba;
    }
  }
  return std::nullopt;
}

bool isFuncArg(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg) {
    return false;
  }
  return isa<func::FuncOp>(arg.getParentBlock()->getParentOp());
}

template <typename T>
bool hasSameType(Value lhs, Value rhs) {
  if constexpr (std::is_same_v<T, BlockArgument>) {
    return isa<T>(lhs) && isa<T>(rhs);
  } else {
    auto lhsDefOp = lhs.getDefiningOp<T>();
    auto rhsDefOp = rhs.getDefiningOp<T>();
    return lhsDefOp != nullptr && rhsDefOp != nullptr;
  }
}

template <typename T>
bool isEqualSlice(T lhs, T rhs) {
  if (lhs == rhs) {
    return true;
  }
  bool sameOffsets = isEqualConstantIntOrValueArray(lhs.getMixedOffsets(),
                                                    rhs.getMixedOffsets());
  bool sameSizes =
      isEqualConstantIntOrValueArray(lhs.getMixedSizes(), rhs.getMixedSizes());
  bool sameStrides = isEqualConstantIntOrValueArray(lhs.getMixedStrides(),
                                                    rhs.getMixedStrides());
  return sameOffsets && sameSizes && sameStrides;
}

bool hasSameIteration(Value lhs, Value rhs);

bool hasSameIteration(memref::SubViewOp lhs, memref::SubViewOp rhs) {
  if (!isEqualSlice(lhs, rhs)) {
    return false;
  }
  return hasSameIteration(lhs.getViewSource(), rhs.getViewSource());
}

bool hasSameIteration(tensor::ExtractSliceOp lhs, tensor::ExtractSliceOp rhs) {
  if (!isEqualSlice(lhs, rhs)) {
    return false;
  }
  return hasSameIteration(lhs.getSource(), rhs.getSource());
}

bool hasSameIteration(BlockArgument lhs, BlockArgument rhs) {
  if (isFuncArg(lhs) && isFuncArg(rhs)) {
    // successfully reach func arg
    return true;
  }
  if (isFuncArg(lhs) || isFuncArg(rhs)) {
    // only one side is func arg, conservatively return false
    return false;
  }
  auto lhsForOp = dyn_cast<scf::ForOp>(lhs.getParentBlock()->getParentOp());
  auto rhsForOp = dyn_cast<scf::ForOp>(rhs.getParentBlock()->getParentOp());
  if (!lhsForOp || !rhsForOp) {
    // TODO: will there be other cases like Forall?
    // TODO: can lhs and rhs come from different for loops?
    return false;
  }

  std::optional<Value> lhsInit = getLoopInitArg(lhsForOp, lhs);
  std::optional<Value> rhsInit = getLoopInitArg(rhsForOp, rhs);
  if (!lhsInit.has_value() || !rhsInit.has_value()) {
    return false;
  }
  return hasSameIteration(lhsInit.value(), rhsInit.value());
}

bool hasSameIteration(Value lhs, Value rhs) {
  LDBG("- check same iteration for: ");
  LDBG("-- lhs: " << lhs);
  LDBG("-- rhs: " << rhs);
  if (hasSameType<BlockArgument>(lhs, rhs)) {
    return hasSameIteration(llvm::cast<BlockArgument>(lhs),
                            llvm::cast<BlockArgument>(rhs));
  }
  if (hasSameType<memref::SubViewOp>(lhs, rhs)) {
    auto lhsOp = lhs.getDefiningOp<memref::SubViewOp>();
    auto rhsOp = rhs.getDefiningOp<memref::SubViewOp>();
    return hasSameIteration(lhsOp, rhsOp);
  }
  if (hasSameType<tensor::ExtractSliceOp>(lhs, rhs)) {
    auto lhsOp = lhs.getDefiningOp<tensor::ExtractSliceOp>();
    auto rhsOp = rhs.getDefiningOp<tensor::ExtractSliceOp>();
    return hasSameIteration(lhsOp, rhsOp);
  }
  // TODO: is there any other related op for check same iteration
  LDBG("- unsupported op for check same iteration");
  return false;
}

bool hasSameIteration(vector::TransferReadOp read,
                      vector::TransferWriteOp write) {
  LDBG("- check same iteration for: ");
  LDBG("-- read: " << read);
  LDBG("-- write: " << write);
  assert(read->getBlock() == write->getBlock() &&
         "read and write must be in the same block");
  Value readV = read.getSource();
  Value writeV = write.getSource();
  return hasSameIteration(readV, writeV);
}

// `read` arg is inplace resuable for `write` arg if
// 1. `write` arg is only written once
// 2. `read` arg is only read once
// 3. `read` op is in the same block as `write` op
// 4. `read` op must precedes `write` op
// 5. `read` op should have same subview iterations as `write` op
void VFInplaceReuseAnalysis::buildVFCalleeFunc(func::FuncOp vfCalleeFunc) {
  ReadMapT arg2reads;
  WriteMapT arg2writes;
  traceFuncArgReadsAndWrites(vfCalleeFunc, arg2reads, arg2writes);

  auto onlyWriteOnce = [&](BlockArgument ba) -> bool {
    return arg2writes[ba].size() == 1 && arg2reads[ba].empty();
  };
  auto onlyReadOnce = [&](BlockArgument ba) -> bool {
    return arg2reads[ba].size() == 1 && arg2writes[ba].empty();
  };

  VFArgInfoMapT argInfo;
  for (BlockArgument ba : vfCalleeFunc.getArguments()) {
    if (!onlyWriteOnce(ba)) {
      continue;
    }
    auto write = cast<vector::TransferWriteOp>(arg2writes[ba].front());
    SetVector<Value> reusableReadArgs;
    llvm::for_each(
        getPrevReadsInSameBlock(write), [&](vector::TransferReadOp read) {
          auto readArg = getReadArg(read, arg2reads);
          if (!readArg.has_value() || !onlyReadOnce(readArg.value()) ||
              !hasSameIteration(read, write)) {
            return;
          }
          reusableReadArgs.insert(readArg.value());
        });
    argInfo[ba] = reusableReadArgs;
  }
  callee2info[vfCalleeFunc] = argInfo;
}

void VFInplaceReuseAnalysis::buildVFCallerFunc(func::FuncOp callerFunc) {
  mlir::SymbolTableCollection symbolTable;
  callerFunc.walk([&](func::CallOp callOp) {
    if (!hivm::isVFCall(callOp)) {
      return WalkResult::skip();
    }
    SymbolRefAttr calleeName = callOp.getCalleeAttr();
    auto calleeFunc =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(callOp, calleeName);
    if (!calleeFunc) {
      LDBG("callee func not found for call op: " << callOp);
      return WalkResult::skip();
    }
    if (!callee2info.contains(calleeFunc)) {
      return WalkResult::skip();
    }
    if (!caller2info.contains(callerFunc)) {
      caller2info[callerFunc] = VFCallInplaceReuseInfo(callerFunc);
    }
    auto &callerInfo = caller2info[callerFunc];
    callerInfo.update(callOp, callee2info[calleeFunc]);
    return WalkResult::advance();
  });
}

/// Gets inplace-reuse info (if any) for the callOp inside given funcOp.
VFCallInplaceReuseInfo *
VFInplaceReuseAnalysis::getVFCallInplaceReuseInfo(func::FuncOp funcOp) {
  auto it = caller2info.find(funcOp);
  return it == caller2info.end() ? nullptr : &it->second;
}

/// Dumps the inplace-reuse information in a human readable format.
void VFInplaceReuseAnalysis::dump() const { print(llvm::errs()); }

/// Dumps the inplace-reuse information to the given stream.
void VFInplaceReuseAnalysis::print(raw_ostream &os) const {
  os << "-------- print VFInplaceReuseAnalysis info ---------\n";
  for (auto [caller, info] : caller2info) {
    info.print(os);
  }
}

/// Returns true if the given gen and kill buffer pair is inplace reusable.
bool VFCallInplaceReuseInfo::isInplaceReusable(Operation *op, Value genBuffer,
                                               Value killBuffer) {
  if (!hivm::isVFCall(op)) {
    return false;
  }
  auto callOp = cast<func::CallOp>(op);
  if (!callsite2bufferinfo.contains(callOp)) {
    return false;
  }

  VFArgInfoMapT replaceInfo = callsite2bufferinfo[callOp];
  if (!replaceInfo.contains(genBuffer)) {
    return false;
  }
  VFArgListT replaceList = replaceInfo[genBuffer];
  return replaceList.contains(killBuffer);
}

Value traceAllocBuffer(Value value) {
  std::optional<memref::AllocOp> alloc = utils::tracebackMemRefToAlloc(value);
  if (!alloc.has_value()) {
    // Conservatively returns original value if fails to trace alloc buffer.
    // TODO: unnecessary because vf call arguments must be alloc on ub.
    return value;
  }
  return alloc.value();
}

SmallVector<Value>
VFCallInplaceReuseInfo::getInplaceReusableOperands(Operation *op,
                                                   Value operand) {
  SetVector<Value> operands(op->operand_begin(), op->operand_end());
  if (!operands.contains(operand)) {
    return {};
  }
  operands.remove(operand);

  SmallVector<Value> reusable;
  for (Value candidate : operands) {
    // trace operand to the buffer to check inplace reusable
    Value buffer1 = traceAllocBuffer(operand);
    Value buffer2 = traceAllocBuffer(candidate);
    if (isInplaceReusable(op, buffer1, buffer2) ||
        isInplaceReusable(op, buffer2, buffer1)) {
      reusable.push_back(candidate);
    }
  }
  return reusable;
}

void VFCallInplaceReuseInfo::update(func::CallOp callOp,
                                    const VFArgInfoMapT &calleeInfo) {
  VFArgInfoMapT bufferInfo;
  for (auto [calleeArg, calleeReuseList] : calleeInfo) {
    unsigned calleeArgIdx = llvm::cast<BlockArgument>(calleeArg).getArgNumber();
    Value callsiteArg = callOp->getOperand(calleeArgIdx);
    Value argBuffer = traceAllocBuffer(callsiteArg);
    VFArgListT reuseBuffers;
    for (Value calleeReuseArg : calleeReuseList) {
      unsigned calleeReuseIdx =
          llvm::cast<BlockArgument>(calleeReuseArg).getArgNumber();
      Value reuseBuffer = traceAllocBuffer(callOp.getOperand(calleeReuseIdx));
      reuseBuffers.insert(reuseBuffer);
    }
    bufferInfo[argBuffer] = reuseBuffers;
  }
  callsite2bufferinfo[callOp] = bufferInfo;
}

/// Dumps the vf call inplace-reuse information in a human readable format.
void VFCallInplaceReuseInfo::dump() const { print(llvm::errs()); }

/// Dumps the vf call inplace-reuse information to the given stream.
void VFCallInplaceReuseInfo::print(raw_ostream &os) const {
  os << "-------- VFCallInplaceReuseInfo ---------\n";
  os << "vf caller func: " << callerFunc << "\n";
  for (auto [call, info] : callsite2bufferinfo) {
    os << "- vf callsite to buffer info:\n";
    os << "- vf call: " << call << "\n";
    for (auto [operand, reuseList] : info) {
      os << "-- inplace reusable values for operand: " << operand << "\n";
      for (Value reuseV : reuseList) {
        os << "--- " << reuseV << "\n";
      }
    }
  }
}

} // namespace hivm
} // namespace mlir