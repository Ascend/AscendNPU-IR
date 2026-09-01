//===- DialectExtension.cpp - Bishengir transform dialect extension -------===//
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

#include "bishengir/Dialect/Transform/IR/TransformOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class BishengirTransformDialectExtension
    : public transform::TransformDialectExtension<
          BishengirTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<func::FuncDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/Transform/IR/TransformOps.cpp.inc"
        >();
  }
};
} // namespace

void bishengir::transform::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<BishengirTransformDialectExtension>();
}
