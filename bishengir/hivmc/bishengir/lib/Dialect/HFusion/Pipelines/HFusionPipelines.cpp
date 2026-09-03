//===- HFusionPipelines.cpp - HFusion pipelines -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Conversion/ArithToAffine/ArithToAffine.h"
#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/Passes.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/Pipelines/Passes.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/MemRef/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Symbol/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "hfusion-pipeline"

namespace mlir {
namespace hfusion {

enum CanonicaliziationPattern {
  FoldFillWithTensorReshapeCollapse = 0,
  FoldFillWithTensorReshapeExpand = 1,
  FoldTransposeWithTranspose = 2
};

static DenseMap<int, std::string> canonicalizationEnumMap = {
    {FoldFillWithTensorReshapeCollapse,
     "(anonymous "
     "namespace)::FoldFillWithTensorReshape<mlir::tensor::CollapseShapeOp>"},
    {FoldFillWithTensorReshapeExpand,
     "(anonymous "
     "namespace)::FoldFillWithTensorReshape<mlir::tensor::ExpandShapeOp>"},
    {FoldTransposeWithTranspose, "FoldTransposeWithTranspose"}};

enum DisableCanonicalizationPhase {
  NoRestriction = 0,
  AfterFlattenBeforeAutoSchedule = 1,
  AfterAutoSchedule = 2
};

static DenseMap<int, std::vector<std::string>> phaseToDisabledMap = {
    {NoRestriction, {}},
    {AfterFlattenBeforeAutoSchedule,
     {canonicalizationEnumMap[FoldFillWithTensorReshapeCollapse],
      canonicalizationEnumMap[FoldFillWithTensorReshapeExpand]}},
    {AfterAutoSchedule, {canonicalizationEnumMap[FoldTransposeWithTranspose]}}};


} // namespace hfusion
} // namespace mlir
