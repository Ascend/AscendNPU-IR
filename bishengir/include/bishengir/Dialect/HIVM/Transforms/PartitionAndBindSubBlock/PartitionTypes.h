//------------------------------PartitionTypes.h------------------------------//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_PARTITIONTYPES_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_PARTITIONTYPES_H

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace hivm {
namespace partition_and_bind {

inline constexpr llvm::StringLiteral kSubBlockAttrName = "sub_block";

inline constexpr llvm::StringLiteral kSubBlockBoundOpAttrName =
    "already_sub_block_bound";

/// UnitAttr tag stamped on a fixpipe whose post-cube result is needed on BOTH
/// sub-blocks, so it must deposit the cube result into both sub-cores' UB The
/// double-write CODEGEN is not yet implemented; the partition pass only records
/// the requirement
inline constexpr llvm::StringLiteral kFixpipeDoubleWriteAttrName =
    "hivm.fixpipe_double_write";

inline constexpr int64_t kSubBlockDim = 0;

inline constexpr unsigned kNumCores = 2;

///     Bottom
///       /  \
///      V0   V1
///       \  /
///      Both
enum class Core : uint8_t {
  Bottom = 0,
  V0 = 1,
  V1 = 2,
  Both = 3,
};

inline Core join(Core a, Core b) {
  if (a == Core::Bottom)
    return b;
  if (b == Core::Bottom)
    return a;
  if (a == b)
    return a;
  return Core::Both;
}

inline bool isSingleCore(Core l) { return l == Core::V0 || l == Core::V1; }

/// Map a single core to its sub-block integer (V0->0, V1->1). Returns nullopt
/// for Bottom/Both. The inverse of `coreFromIndex`.
inline std::optional<int64_t> coreIndex(Core l) {
  switch (l) {
  case Core::V0:
    return 0;
  case Core::V1:
    return 1;
  default:
    return std::nullopt;
  }
}

/// Map a sub-block integer (0/1) to its single Core; out-of-range -> Bottom.
inline Core coreFromIndex(int64_t idx) {
  if (idx == 0)
    return Core::V0;
  if (idx == 1)
    return Core::V1;
  return Core::Bottom;
}

//===----------------------------------------------------------------------===//
// Supernode
//===----------------------------------------------------------------------===//

struct Supernode {
  /// The `{sub_block = n}` scope that anchors this unit.
  scope::ScopeOp outerScope;

  Core core = Core::Bottom;

  Supernode() = default;
  Supernode(scope::ScopeOp outer, Core l) : outerScope(outer), core(l) {}
};

bool isCubeOrSharedOp(Operation *op);

/// If `op` is a `scope.scope` carrying `{sub_block = n : i64}`, return its Core
/// (V0/V1). Returns Bottom when the attr is absent
Core getSubBlockCoreOf(Operation *op);

bool isOperandParallelSubBlockGuard(Operation *op);

} // namespace partition_and_bind
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_PARTITIONANDBINDSUBBLOCK_PARTITIONTYPES_H