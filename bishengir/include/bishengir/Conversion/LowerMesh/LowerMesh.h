//===- LowerMesh.h - Mesh to Collectives ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the HACC dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_LOWERMESH_LOWERMESH_H
#define BISHENGIR_CONVERSION_LOWERMESH_LOWERMESH_H

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace bishengir {
namespace mesh {
enum class TargetLib : uint64_t { HCCL = 0, LCCL };
} // namespace mesh
} // namespace bishengir

namespace mlir {

#define GEN_PASS_DECL_LOWERMESH
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<Pass>
createMeshLoweringPass(const LowerMeshOptions &options = {});

} // namespace mlir
#endif
