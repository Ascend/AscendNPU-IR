//===- HIVMDoc.td - HIVM documentation ---------------------*- tablegen -*-===//
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
// This documentation files exists to circumvent limitations on mixing different
// .td files in cases one does not want to have all ops belong to the same
// logical unit. This file should only include other .td files only and be used
// for the purpose of generating documentation.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD

// Ops
include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.td"

// Attributes
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
