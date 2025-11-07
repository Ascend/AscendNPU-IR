//===- MemRefImpl.h - MemRef Dialect Implementation -----------------------===//
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

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace memref {

/// Create memref.alloc op with the same type as source
Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source);

/// Create memref.alloc op with the same shape as source
/// but with element type targetElemType
Value createMemRefAllocOpWithTargetElemType(
    OpBuilder &builder, Location loc, Value source, Type targetElemType,
    std::optional<MemRefLayoutAttrInterface> layout = std::nullopt);

void getExtendedCanonicalizationPatterns(mlir::RewritePatternSet &results);

} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H
