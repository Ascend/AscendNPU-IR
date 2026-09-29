//===----------------------------- SimtScalarCalls.h ------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_UTILS_SIMTSCALARCALLS_H
#define BISHENGIR_DIALECT_HIVM_UTILS_SIMTSCALARCALLS_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hivm {

/// Attribute carrying the shared-memory byte offset of a hoisted scalar call's
/// store, and marking that store as one to rebuild in the SIMD caller.
constexpr llvm::StringLiteral kShmemOffsetStampName = "use_shmem_offset";

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_UTILS_SIMTSCALARCALLS_H
