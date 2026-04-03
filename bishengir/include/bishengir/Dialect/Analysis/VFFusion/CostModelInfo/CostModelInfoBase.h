//===- CostModelInfoBase.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_COSTMODELNFOBASE_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_COSTMODELNFOBASE_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::analysis {
//===----------------------------------------------------------------------===//
// HardwareInfoBase
//===----------------------------------------------------------------------===//
enum class TypeKind { F16, F32, F64, I1, I32, I64, Index, Unknown };
struct CostInfo {
  CostInfo(int execInterval, int execLatency, int execUnit = 2)
      : execInterval(execInterval), execLatency(execLatency), execUnit(execUnit) {}
  int execInterval;
  int execLatency;
  int execUnit;
};
using OpConfigMap =
    llvm::DenseMap<mlir::TypeID, llvm::DenseMap<TypeKind, CostInfo>>;

class CostModelInfoBase {
public:
  virtual ~CostModelInfoBase() = default;
  virtual const OpConfigMap getConfigMap(bool isReduction) const = 0;
};
using VALUInfoPtr = std::unique_ptr<CostModelInfoBase>;
} // namespace mlir::analysis
#endif
