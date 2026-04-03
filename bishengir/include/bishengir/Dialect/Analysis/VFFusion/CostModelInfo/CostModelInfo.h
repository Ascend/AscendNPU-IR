//===- CostModelInfo.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_COSTMODELINFO_H
#define BISHENGIR_DIALECT_ANALYSIS_VFFUSION_COSTMODELINFO_COSTMODELINFO_H
#include "bishengir/Dialect/Analysis/VFFusion/CostModelInfo/CostModelInfoBase.h"
namespace mlir {
namespace analysis {
class CostModelInfo : public CostModelInfoBase {
public:
  const OpConfigMap getConfigMap(bool isReduction) const override {
    return isReduction ? ReductionOpCostInfos : ParallelOpCostInfos;
  }

protected:
  OpConfigMap ParallelOpCostInfos = {
      // ADD OP
      {mlir::TypeID::get<mlir::arith::AddFOp>(),
       {{TypeKind::F32, CostInfo{1, 4}},
        {TypeKind::F16, CostInfo{1, 4}}}},
      {mlir::TypeID::get<mlir::arith::AddIOp>(),
       {{TypeKind::I32, CostInfo{2, 4}}}},
      // CAST OP
      {mlir::TypeID::get<mlir::arith::SIToFPOp>(),
       {{TypeKind::F32, CostInfo{2, 5}}}},
      {mlir::TypeID::get<mlir::arith::TruncFOp>(),
       {{TypeKind::F16, CostInfo{1, 4}}}},
      {mlir::TypeID::get<mlir::arith::ExtFOp>(),
       {{TypeKind::F32, CostInfo{1, 4}}}},
      {mlir::TypeID::get<mlir::arith::UIToFPOp>(),
       {{TypeKind::F32, CostInfo{0, 0}}}},
      {mlir::TypeID::get<mlir::arith::FPToSIOp>(),
       {{TypeKind::I64, CostInfo{2, 5}}}},
      // CMP OP
      {mlir::TypeID::get<mlir::arith::CmpFOp>(),
       {{TypeKind::F32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::CmpIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      // MUL OP
      {mlir::TypeID::get<mlir::arith::MulFOp>(),
       {{TypeKind::F16, CostInfo{1, 5}},
        {TypeKind::F32, CostInfo{1, 5}}}},
      {mlir::TypeID::get<mlir::arith::MulIOp>(),
       {{TypeKind::I32, CostInfo{2, 5}}}},
      // SUB OP
      {mlir::TypeID::get<mlir::arith::SubFOp>(),
       {{TypeKind::F32, CostInfo{1, 4}}}},
      {mlir::TypeID::get<mlir::arith::SubIOp>(),
       {{TypeKind::I32, CostInfo{2, 5}}}},
      // DIV OP
      {mlir::TypeID::get<mlir::arith::DivFOp>(),
       {{TypeKind::F16, CostInfo{8, 19}},
        {TypeKind::F32, CostInfo{4, 14}}}},
      // Math OP
      {mlir::TypeID::get<mlir::math::SqrtOp>(),
       {{TypeKind::F16, CostInfo{8, 19}},
        {TypeKind::F32, CostInfo{4, 14}}}},
      {mlir::TypeID::get<mlir::math::ExpOp>(),
       {{TypeKind::F16, CostInfo{8, 18}},
        {TypeKind::F32, CostInfo{4, 13}}}},
      {mlir::TypeID::get<mlir::math::LogOp>(),
       {{TypeKind::F32, CostInfo{4, 15}}}},
      // AND OP
      {mlir::TypeID::get<mlir::arith::AndIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      // max/min
      {mlir::TypeID::get<mlir::arith::MinimumFOp>(),
       {{TypeKind::F32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::MaximumFOp>(),
       {{TypeKind::F32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::MaxSIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::MinSIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      // Select
      {mlir::TypeID::get<mlir::arith::SelectOp>(),
       {{TypeKind::F32, CostInfo{1, 3}}}},
      // Other
      {mlir::TypeID::get<mlir::arith::ExtUIOp>(),
       {{TypeKind::I32, CostInfo{0, 0}}}},
      {mlir::TypeID::get<mlir::arith::ExtSIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::BitcastOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::linalg::IndexOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::IndexCastOp>(),
       {{TypeKind::Index, CostInfo{0, 0}}}},
      {mlir::TypeID::get<mlir::arith::ShRUIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::ShLIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::XOrIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      {mlir::TypeID::get<mlir::arith::RemSIOp>(),
       {{TypeKind::I32, CostInfo{1, 3}}}},
      };

  OpConfigMap ReductionOpCostInfos{
      // ReductionOp will expand in the future pass
      {mlir::TypeID::get<mlir::arith::MaximumFOp>(),
       {{TypeKind::F32, CostInfo{1, 13, 2}}}},
      {mlir::TypeID::get<mlir::arith::AddFOp>(),
       {{TypeKind::F32, CostInfo{1, 19, 2}}}},
  };
};
} // namespace analysis
} // namespace mlir
#endif
