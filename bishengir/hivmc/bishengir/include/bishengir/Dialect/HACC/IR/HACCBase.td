//===- HACCBase.td - Base definitions for HACC dialect -----*- tablegen -*-===//
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
//
// This is the definition file for base heterogeneous async computing
// call support.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD

include "mlir/IR/OpBase.td"

def HACC_Dialect : Dialect {
  let name = "hacc";
  let description = [{
    Heterogeneous Async Computing Call (HACC) dialect.
  }];
  let cppNamespace = "::mlir::hacc";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "mlir::DLTIDialect"
  ];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
