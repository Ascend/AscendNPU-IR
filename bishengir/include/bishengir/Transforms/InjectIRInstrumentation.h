//===- InjectIRInstrumentation.h - Pass-based IR injection -----*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_TRANSFORMS_INJECTIRINSTRUMENTATION_H
#define BISHENGIR_TRANSFORMS_INJECTIRINSTRUMENTATION_H

#include "mlir/Pass/PassInstrumentation.h"

#include <string>
#include <utility>

namespace bishengir {

/// Replaces the current module from a file before or after a selected pass.
/// Injection specifications use the format: pass-id@file-path.
class InjectIRInstrumentation : public mlir::PassInstrumentation {
public:
  InjectIRInstrumentation(bool printPassId, std::string injectIrBefore,
                          std::string injectIrAfter)
      : printPassId(printPassId),
        injectIrBefore(std::move(injectIrBefore)),
        injectIrAfter(std::move(injectIrAfter)) {}

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

private:
  bool printPassId;
  std::string injectIrBefore;
  std::string injectIrAfter;
};

} // namespace bishengir

#endif // BISHENGIR_TRANSFORMS_INJECTIRINSTRUMENTATION_H
