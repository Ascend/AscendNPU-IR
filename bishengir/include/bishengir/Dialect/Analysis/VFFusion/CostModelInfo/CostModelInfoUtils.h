//===- CostModelInfoUtils.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_UTILS_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_UTILS_H
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfo.h"
#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfoBase.h"
namespace mlir {
namespace analysis {
class CostModelInfoUtils : public CostModelInfoBase {
public:
    static TypeKind  getTypeKind(mlir::Type type);
    static OpConfigMap getOpConfigMap(hacc::TargetDevice dev,
                                  bool isReduction = false);
    static CostInfo getOpCostInfo(mlir::Operation *op, bool isReduction = false);
    static CostInfo lookupConfig(const OpConfigMap &targetMap,
                             mlir::TypeID opTypeID, mlir::Type dataType,
                             const llvm::StringRef opName);                               

};
} // namespace analysis
} // namespace mlir

#endif