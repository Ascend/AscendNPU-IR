//===- MathExtDialect.cpp - MLIR dialect for Math Ext implementation ------===//
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

#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::mathExt;

#include "bishengir/Dialect/MathExt/IR/MathExtOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathExtInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void mlir::mathExt::MathExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/MathExt/IR/MathExtOps.cpp.inc"
      >();
  addInterfaces<MathExtInlinerInterface>();
}
