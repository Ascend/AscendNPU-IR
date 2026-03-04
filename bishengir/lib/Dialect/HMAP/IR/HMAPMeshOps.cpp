//===- HMAPMeshOps.cpp - HMAP specific Mesh Dialect Operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HMAP/IR/HMAP.h"

namespace mlir {
namespace mesh {

//===----------------------------------------------------------------------===//
// Mesh utilities
//===----------------------------------------------------------------------===//

static FailureOr<MeshOp> getMeshAndVerify(Operation *op,
                                          FlatSymbolRefAttr meshSymbol,
                                          SymbolTableCollection &symbolTable) {
  MeshOp mesh = getMeshOrNull(op, meshSymbol, symbolTable);
  if (!mesh) {
    return op->emitError() << "Undefined required mesh symbol \""
                           << meshSymbol.getValue() << "\".";
  }

  return mesh;
}

template <typename It> bool isUnique(It begin, It end) {
  if (begin == end) {
    return true;
  }
  It next = std::next(begin);
  if (next == end) {
    return true;
  }
  for (; next != end; ++next, ++begin) {
    if (*begin == *next) {
      return false;
    }
  }
  return true;
}

static LogicalResult verifyMeshAxes(Location loc, ArrayRef<MeshAxis> axes,
                                    MeshOp mesh) {
  SmallVector<MeshAxis> sorted = llvm::to_vector(axes);
  llvm::sort(sorted);
  if (!isUnique(sorted.begin(), sorted.end())) {
    return emitError(loc) << "Mesh axes contains duplicate elements.";
  }

  MeshAxis rank = mesh.getRank();
  for (auto axis : axes) {
    if (axis >= rank || axis < 0) {
      return emitError(loc)
             << "0-based mesh axis index " << axis
             << " is out of bounds. The referenced mesh \"" << mesh.getSymName()
             << "\" is of rank " << rank << ".";
    }
  }

  return success();
}

template <typename Op>
static FailureOr<MeshOp>
getMeshAndVerifyAxes(Op op, SymbolTableCollection &symbolTable) {
  auto mesh =
      getMeshAndVerify(op.getOperation(), op.getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyMeshAxes(op.getLoc(), op.getMeshAxes(), mesh.value()))) {
    return failure();
  }
  return mesh;
}

} // namespace mesh
} // namespace mlir

namespace mlir {
namespace hmap {

//===----------------------------------------------------------------------===//
// hmap.all_to_all_v op
//===----------------------------------------------------------------------===//

LogicalResult
AllToAllVOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = mesh::getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }

  // TODO: Look into verification of alltoall-v
  return success();
}

void AllToAllVOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_to_all_v");
}

} // namespace hmap
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HMAP/IR/HMAPOps.cpp.inc"
