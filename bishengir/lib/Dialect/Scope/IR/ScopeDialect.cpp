//===- ScopeDialect.cpp - Implementation of Scope dialect and types -------===//
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

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

// For function inliner support
#include "mlir/Transforms/InliningUtils.h"

#define GET_OP_CLASSES
#include "bishengir/Dialect/Scope/IR/ScopeOps.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with scope
/// operations. All scope operations can be inlined.
struct ScopeInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
#ifndef NDEBUG
    auto yieldOp = llvm::cast<mlir::scope::ReturnOp>(op);
    assert(yieldOp->getNumOperands() == 0 &&
           "need to update this function if ReturnOp returns operands");
#endif
  }
};

} // namespace

void mlir::scope::ScopeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/Scope/IR/ScopeOps.cpp.inc"
      >();

  // Add function inliner interfaces
  addInterfaces<ScopeInlinerInterface>();
}

#include "bishengir/Dialect/Scope/IR/ScopeDialect.cpp.inc"
