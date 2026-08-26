//===- TestDialect.cpp - MLIR Dialect for Testing BiShengIR  ----*- C++ -*-===//
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

#include "TestDialect.h"
#include "InitTestDialect.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/TypeUtilities.h"

// Include this before the using namespace lines below to
// test that we don't have namespace dependencies.
#include "TestOpsDialect.cpp.inc"

namespace bishengir_test {

void TestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TestOps.cpp.inc"
      >();
}

void registerTestDialect(::mlir::DialectRegistry &registry) {
  registry.insert<TestDialect>();
}

} // namespace bishengir_test

#define GET_OP_CLASSES
#include "TestOps.cpp.inc"
