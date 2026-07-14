//===- RegBaseArchUtils.h ----------------------------------------*- C++ -*-===//
// TODO-A5: Remove this file after appropriate porting of all dependencies
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#pragma once

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace mlir::hfusion {

Operation *createVandOp(OpBuilder &b, Location loc, Value lhs, Value rhs);
Operation *createLinalgElemwiseBinaryOp(OpBuilder &b, Location loc,
                                        linalg::BinaryFn fn, Value lhs,
                                        Value rhs);
Operation *createHFusionElemwiseBinaryOp(OpBuilder &b, Location loc,
                                         hfusion::BinaryFn fn, Value lhs,
                                         Value rhs);
Operation *createCmpOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                       CompareFn cmpFn);

template <typename T> T selectRoundMode(Type inType, Type outType);

} // namespace mlir::hfusion
