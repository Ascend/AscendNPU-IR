//===- MemRefExtDialect.cpp -----------------------------------------------===//
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

#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include <optional>

using namespace mlir;
using namespace bishengir::memref_ext;

#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOpsDialect.cpp.inc"

void bishengir::memref_ext::MemRefExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "bishengir/Dialect/MemRefExt/IR/MemRefExtOps.cpp.inc"
      >();
  declarePromisedInterface<ConvertToLLVMPatternInterface, MemRefExtDialect>();
}
