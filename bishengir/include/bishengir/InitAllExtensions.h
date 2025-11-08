//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
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
//
// This file defines a helper to trigger the registration of all bishengir
// dialect extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INITALLEXTENSIONS_H_
#define BISHENGIR_INITALLEXTENSIONS_H_

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/TransformOps/HFusionTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
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
