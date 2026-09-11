//===- FixpipePreQuantContract.h - pre-quant type contract ------*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_FIXPIPEPREQUANTCONTRACT_H
#define BISHENGIR_DIALECT_HIVM_IR_FIXPIPEPREQUANTCONTRACT_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace hivm {
// Forward declaration matching the generated HIVM enum header (included via
// HIVM.h); the definition lives in HIVMEnums.h.inc.
enum class FixpipePreQuantMode : uint32_t;
} // namespace hivm
} // namespace mlir

#include "bishengir/Dialect/HIVM/IR/Contracts/FixpipePreQuantContract.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_FIXPIPEPREQUANTCONTRACT_H
