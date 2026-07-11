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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::hivm::syncsolver {

struct FuncArgInfo {
  func::FuncOp funcOp{nullptr};
  BlockArgument funcArg{nullptr};
  int64_t argNum{-1};
  bool isWorkSpace{false};

  FuncArgInfo() = default;
  explicit FuncArgInfo(func::FuncOp funcOp, BlockArgument funcArg)
      : funcOp(funcOp), funcArg(funcArg), argNum(funcArg.getArgNumber()) {
    assert(funcOp != nullptr && "funcOp is nullptr");
    assert(funcArg != nullptr && "blockArg is nullptr");
    assert(argNum >= 0 && "argNum is negative");
  }

  bool operator==(const FuncArgInfo &other) const {
    return std::tie(funcOp, funcArg, argNum) ==
           std::tie(other.funcOp, other.funcArg, other.argNum);
  }
  bool operator!=(const FuncArgInfo &other) const { return !(*this == other); }

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
  bool isTightlyCoupledBuffer{false};

  PointerLikeInfo() = default;
  explicit PointerLikeInfo(Operation *op) : op(op) {}

  bool operator==(const PointerLikeInfo &other) const { return op == other.op; }
  bool operator!=(const PointerLikeInfo &other) const {
    return !(*this == other);
  }

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

struct AllocLikeInfo {
  Operation *op{nullptr};
  bool isWorkSpace{false};
  bool isTightlyCoupledBuffer{false};

  AllocLikeInfo() = default;
  explicit AllocLikeInfo(Operation *op) : op(op) {}

  bool operator==(const AllocLikeInfo &other) const { return op == other.op; }
  bool operator!=(const AllocLikeInfo &other) const {
    return !(*this == other);
  }

  std::string str();

  static std::optional<AllocLikeInfo> tryGet(memref::AllocOp pointerCastOp);

  static std::optional<AllocLikeInfo> tryGet(Value value);

  static bool checkConflict(const AllocLikeInfo &allocLikeInfo1,
                            const AllocLikeInfo &allocLikeInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

struct MemInfo {
  Value value;
  std::optional<FuncArgInfo> funcArgInfo;
  std::optional<PointerLikeInfo> pointerLikeInfo;
  std::optional<AllocLikeInfo> allocLikeInfo;
  std::optional<PIPE> pipe;

  MemInfo() = default;

  explicit MemInfo(Value value, std::optional<PIPE> pipe = {})
      : value(value), pipe(pipe) {}

  explicit MemInfo(Value value, FuncArgInfo funcArgInfo,
                   std::optional<PIPE> pipe = {})
      : value(value), funcArgInfo(funcArgInfo), pipe(pipe) {}

  explicit MemInfo(Value value, PointerLikeInfo pointerLikeInfo,
                   std::optional<PIPE> pipe = {})
      : value(value), pointerLikeInfo(pointerLikeInfo), pipe(pipe) {}

  explicit MemInfo(Value value, AllocLikeInfo allocLikeInfo,
                   std::optional<PIPE> pipe = {})
      : value(value), allocLikeInfo(allocLikeInfo), pipe(pipe) {}

  bool operator==(const MemInfo &other) const {
    return std::tie(value, funcArgInfo, pointerLikeInfo, allocLikeInfo, pipe) ==
           std::tie(other.value, other.funcArgInfo, other.pointerLikeInfo,
                    other.allocLikeInfo, other.pipe);
  }
  bool operator!=(const MemInfo &other) const { return !(*this == other); }

  std::string str();

  int64_t getSz() const {
    if (pointerLikeInfo.has_value()) {
      return pointerLikeInfo->addresses.size();
    }
    if (value != nullptr) {
      return 1;
    }
    return 0;
  }

  static MemInfo getMemInfo(Value val, std::optional<PIPE> pipe = {});

  static MemInfo getMemInfo(const llvm::SmallVector<int64_t> &addrs);

  static bool checkConflict(const MemInfo &memInfo1, const MemInfo &memInfo2,
                            std::optional<int64_t> lcmLen = {},
                            std::optional<int64_t> eventIdNum = {});
};

} // namespace mlir::hivm::syncsolver

#endif // BISHENG_DIALECT_HIVM_TRANSFORMS_GRAPHSYNCSOLVER_MEMINFO_H
