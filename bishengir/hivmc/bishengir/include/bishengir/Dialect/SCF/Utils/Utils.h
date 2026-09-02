//===-----------------------Utils.h----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
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
