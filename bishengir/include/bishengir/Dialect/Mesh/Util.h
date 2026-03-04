//===- Util.h - Utility functions for Mesh dialect related passes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_MESH_UTIL_H
#define BISHENGIR_DIALECT_MESH_UTIL_H

#include "mlir/IR/TypeRange.h"
namespace llvm {
class StringRef;
}
namespace mlir {
class ModuleOp;
class Location;
class OpBuilder;

namespace func {
class FuncOp;
} // namespace func

} // namespace mlir
namespace bishengir {
mlir::func::FuncOp getCustomFunction(llvm::StringRef name,
                                     mlir::ModuleOp parent, mlir::Location loc,
                                     mlir::OpBuilder &rewriter,
                                     mlir::TypeRange funcArgs,
                                     mlir::TypeRange results = {});
} // namespace bishengir

#endif // BISHENGIR_DIALECT_MESH_UTIL_H
