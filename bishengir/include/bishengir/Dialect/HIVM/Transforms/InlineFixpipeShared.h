//===- InlineFixpipeShared.h - Shared InlineFixpipe queries ---------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPESHARED_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPESHARED_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <type_traits>

namespace mlir::hivm {

inline constexpr llvm::StringLiteral kFallbackNotExec = "fallback_not_exec";

bool isInsideVectorScope(Operation *op);
Operation *getSingleSiftedUser(Value v);
SmallVector<VCastOp> collectVCastChain(VCastOp firstCast);
std::optional<FixpipePreQuantMode>
getQuantModeForCastChain(ArrayRef<VCastOp> castChain);
std::optional<FixpipePreQuantMode> getQuantMode(hivm::VCastOp castOp);
bool isUserTransposeInlinable(FixpipeOp op, Operation *user);

template <typename OpType>
std::optional<FixpipePreReluMode> getReluMode(OpType op) {
  if constexpr (std::is_same_v<OpType, hivm::VReluOp>)
    return hivm::symbolizeFixpipePreReluMode("NORMAL_RELU");
  llvm::report_fatal_error("unsupported ReluValue");
}

} // namespace mlir::hivm

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_INLINEFIXPIPESHARED_H
