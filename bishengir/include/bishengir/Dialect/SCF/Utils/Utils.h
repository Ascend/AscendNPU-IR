//===-----------------------Utils.h----------------------------------------===//
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

#ifndef BISHENGIR_DIALECT_SCF_UTILS_UTILS_H
#define BISHENGIR_DIALECT_SCF_UTILS_UTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace scf {
namespace utils {

DiagnosedSilenceableFailure
mapForToForallImpl(OpBuilder &builder, scf::ForOp forOp,
                   std::optional<ArrayAttr> deviceMappings,
                   scf::ForallOp &forallOp, bool CheckMapForToForall = false);

bool isNormalized(LoopLikeOpInterface forOp);

} // namespace utils
} // namespace scf
} // namespace mlir

#endif // BISHENGIR_DIALECT_SCF_UTILS_UTILS_H
