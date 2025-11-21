//===- Scope.h - Scope Dialect -----------------------------------*- C++-*-===//
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

#ifndef BISHENGIR_DIALECT_SCOPE_IR_SCOPE_H
#define BISHENGIR_DIALECT_SCOPE_IR_SCOPE_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Scope Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Scope/IR/ScopeDialect.h.inc"

//===----------------------------------------------------------------------===//
// Scope Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Scope/IR/ScopeOps.h.inc"

#endif // BISHENGIR_DIALECT_SCOPE_IR_SCOPE_H