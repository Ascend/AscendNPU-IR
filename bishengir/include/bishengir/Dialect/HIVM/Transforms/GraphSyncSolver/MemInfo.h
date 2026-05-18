//===------------- MemInfo.h ---- Graph Sync Solver -----------------------===//
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
#ifndef BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H
#define BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include <climits>
#include <pthread.h>

namespace mlir::hivm::syncsolver {

struct FuncArgInfo {
  func::FuncOp funcOp{nullptr};
  BlockArgument funcArg{nullptr};
  int64_t argNum{-1};
  bool isWorkSpace{false};

  FuncArgInfo() = default;
  explicit FuncArgInfo(func::FuncOp funcOp, BlockArgument funcArg,
                       bool isWorkSpace = false)
      : funcOp(funcOp), funcArg(funcArg), isWorkSpace(isWorkSpace) {
    assert(funcOp != nullptr && "funcOp is nullptr");
    assert(funcArg != nullptr && "blockArg is nullptr");
    argNum = funcArg.getArgNumber();
  }

  std::string str();

  static std::optional<FuncArgInfo> tryGet(Value value);

  static bool checkConflict(const FuncArgInfo &funcArgInfo1,
                            const FuncArgInfo &funcArgInfo2);
};

struct PointerLikeInfo {
  Operation *op{nullptr};
  llvm::SmallVector<int64_t> addresses;
  std::optional<int64_t> allocateSize;
  std::optional<hivm::AddressSpace> addressSpace;
  LoopLikeOpInterface parentLoop{nullptr};
  bool isWorkSpace{false};

  PointerLikeInfo() = default;
  explicit PointerLikeInfo(Operation *op) : op(op) {}

  std::string str();

  static std::optional<PointerLikeInfo>
  tryGet(hivm::PointerCastOp pointerCastOp);

  static std::optional<PointerLikeInfo>
  tryGet(bishengir::memref_ext::AllocWorkspaceOp allocWorkspaceOp);

  static std::optional<PointerLikeInfo> tryGet(Value value);

  static bool checkConflict(const PointerLikeInfo &pointerLikeInfo1,
                            const PointerLikeInfo &pointerLikeInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

struct MemInfo {
  Value value{nullptr};
  std::optional<FuncArgInfo> funcArgInfo;
  std::optional<PointerLikeInfo> pointerLikeInfo;

  MemInfo() = default;

  explicit MemInfo(Value value) : value(value) {}

  explicit MemInfo(Value value, FuncArgInfo funcArgInfo)
      : value(value), funcArgInfo(funcArgInfo) {}

  explicit MemInfo(Value value, PointerLikeInfo pointerLikeInfo)
      : value(value), pointerLikeInfo(pointerLikeInfo) {}

  int64_t getSz() const {
    if (pointerLikeInfo.has_value()) {
      return pointerLikeInfo->addresses.size();
    }
    if (value != nullptr) {
      return 1;
    }
    return 0;
  }

  std::string str();

  static bool checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

llvm::SmallVector<int64_t> getAddresses(const llvm::SmallVector<Value> &addrs);

PointerLikeInfo getPointerLikeInfo(hivm::PointerCastOp pointerCastOp);

PointerLikeInfo
getPointerLikeInfo(bishengir::memref_ext::AllocWorkspaceOp allocWorkspaceOp);

MemInfo getMemInfo(Value val);

MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs);

bool isWorkSpaceFuncArgument(func::FuncOp funcOp, BlockArgument funcArg);

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H
