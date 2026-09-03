//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all bishengir
// dialect extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INITALLEXTENSIONS_H_
#define BISHENGIR_INITALLEXTENSIONS_H_

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "bishengir/Dialect/Utils/OpInterfaceUtils.h"
#include "mlir/IR/DialectRegistry.h"

namespace bishengir {

inline void registerAllExtensions(mlir::DialectRegistry &registry) {
  // Register all transform dialect extensions.
  mlir::hivm::registerTransformDialectExtension(registry);
  mlir::hacc::func_ext::registerHACCDialectExtension(registry);
  mlir::hacc::llvm_ext::registerHACCDialectExtension(registry);
  mlir::hfusion::registerTransformDialectExtension(registry);
  bishengir::scf::registerTransformDialectExtension(registry);
  mlir::registerOpInterfaceExtensions(registry);
}

} // namespace bishengir

#endif // BISHENGIR_INITALLEXTENSIONS_H_
