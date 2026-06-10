//===- Verify.h - Auto vectorization verification helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_VERIFY_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_VERIFY_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;

namespace hfusion {

class AutoVectorizeVerifier {
public:
  AutoVectorizeVerifier &verifyFreeVectorRegion(bool enabled = true) {
    enabledStrategies.verifyFreeVectorRegion = enabled;
    return *this;
  }

  AutoVectorizeVerifier &verifyFreeVectorFunc(bool enabled = true) {
    enabledStrategies.verifyFreeVectorFunc = enabled;
    return *this;
  }

  AutoVectorizeVerifier &emitDiagnostics(bool enabled = true) {
    emitDiagnosticsFlag = enabled;
    return *this;
  }

  LogicalResult check(Operation *op) const;

private:
  struct EnabledStrategies {
    bool verifyFreeVectorRegion = false;
    bool verifyFreeVectorFunc = false;
  } enabledStrategies;

  bool emitDiagnosticsFlag = true;
};

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_AUTOVECTORIZE_VERIFY_H
