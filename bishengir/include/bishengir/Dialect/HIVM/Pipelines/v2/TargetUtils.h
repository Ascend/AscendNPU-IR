//===- TargetUtils.h - v2 pipeline target capabilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_PIPELINES_V2_TARGETUTILS_H
#define BISHENGIR_DIALECT_HIVM_PIPELINES_V2_TARGETUTILS_H

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Target/NPU/NPUArchitectureFeatures.h"

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir {
namespace hivm {
namespace v2 {

/// The v2 pipeline needs reg-based vector execution and on-chip Cube/Vector
/// data exchange. This covers the A5 and A6 architecture profiles.
inline bool isSupportedTarget(hacc::TargetDevice target) {
  auto architecture = hacc::getArchitecture(target);
  return bishengir::npu::getExecutionModel(architecture) ==
             bishengir::npu::ExecutionModel::kRegBased &&
         bishengir::npu::getCubeVectorDataExchange(architecture) ==
             bishengir::npu::CubeVectorDataExchange::kOnChipSupported;
}

/// The target's default AIC:AIV ratio from its spec, reduced.
inline std::optional<std::pair<int32_t, int32_t>>
getCoreRatio(hacc::TargetDevice target) {
  return hacc::utils::getDefaultCoreRatio(target);
}

/// CV 1:2 needs operand-parallel sub-block partitioning; 1:1 targets skip it.
inline bool needsCV12(hacc::TargetDevice target) {
  auto ratio = getCoreRatio(target);
  return ratio && ratio->first == 1 && ratio->second == 2;
}

} // namespace v2
} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_PIPELINES_V2_TARGETUTILS_H
