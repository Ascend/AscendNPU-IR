//===- VFInplaceReuseAnalyzer.h ---------------------------------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_VF_INPLACE_REUSE_ANALYZER_H
#define BISHENGIR_DIALECT_HIVM_VF_INPLACE_REUSE_ANALYZER_H

#include "bishengir/Dialect/HIVM/Utils/RegbaseUtils.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hivm {

// value list for vf arguments/parameters
using VFArgListT = SetVector<Value>;
// map from `write` value to inplace reusable `read` values
using VFArgInfoMapT = DenseMap<Value, VFArgListT>;
// vf value inplace reuse information at each callsite
class VFCallInplaceReuseInfo;

class VFInplaceReuseAnalysis {
public:
  using VFCalleeFuncMapT = DenseMap<func::FuncOp, VFArgInfoMapT>;
  using VFCallerFuncMapT = DenseMap<func::FuncOp, VFCallInplaceReuseInfo>;

public:
  /// Creates a new vf-inplace-reuse analysis that computes argument inplace
  /// reuse information for all associated vf functions and vf calls.
  explicit VFInplaceReuseAnalysis(ModuleOp moduleOp);

  /// Gets inplace-reuse info (if any) for the vf call inside given funcOp.
  VFCallInplaceReuseInfo *getVFCallInplaceReuseInfo(func::FuncOp funcOp);

  /// Dumps the inplace-reuse information in a human readable format.
  void dump() const;

  /// Dumps the inplace-reuse information to the given stream.
  void print(raw_ostream &os) const;

private:
  /// Initializes the internal mappings.
  void build();
  void buildVFCalleeFunc(func::FuncOp funcOp);
  void buildVFCallerFunc(func::FuncOp caller);

  // Map func to inplace reuse information for vf callee func
  VFCalleeFuncMapT callee2info;

  // Map func to inplace reuse information for vf caller func
  VFCallerFuncMapT caller2info;

  ModuleOp moduleOp;
};

class VFCallInplaceReuseInfo {
public:
  using VFCallFuncMapT = DenseMap<func::CallOp, VFArgInfoMapT>;

  explicit VFCallInplaceReuseInfo(func::FuncOp caller = nullptr)
      : callerFunc(caller) {}

  /// Returns true if the given gen and kill `buffer` pair is inplace reusable.
  bool isInplaceReusable(Operation *op, Value genBuffer, Value killBuffer);

  /// Returns all possible inplace reusable `op` operands for given `operand`.
  SmallVector<Value> getInplaceReusableOperands(Operation *op, Value operand);

  /// Dumps the vf call inplace-reuse information in a human readable format.
  void dump() const;

  /// Dumps the vf call inplace-reuse information to the given stream.
  void print(raw_ostream &os) const;

  /// Returns true if any two VF operands trace to the same alloc.
  /// If so, skip inplace reuse for this call to avoid unsound optimizations.
  static bool hasAliasArgRisk(Operation *op);

private:
  // Map call site to inplace reuse buffers (i.e. alloc) for caller arguments.
  //
  // We use buffer instead of caller argument iteself, because callsite
  // arguments can be subview from buffer/alloc, while plan-memory handles only
  // buffers, not subviews
  VFCallFuncMapT callsite2bufferinfo;

  // caller func of vf calls.
  func::FuncOp callerFunc;

  // Update call op info with callee func info.
  void update(func::CallOp callOp, const VFArgInfoMapT &calleeInfo);

  friend class VFInplaceReuseAnalysis;
};

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_VF_INPLACE_REUSE_ANALYZER_H
