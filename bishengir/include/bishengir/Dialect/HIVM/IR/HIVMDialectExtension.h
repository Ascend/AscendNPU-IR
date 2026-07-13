//===- HIVMDiallctExtension.h - HIVM Dialect Extension -------------------===//
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

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace bishengir {
namespace hivm {
namespace detail {
void registerLibraryFunctionOpInterfaceExtension(
    mlir::DialectRegistry &registry);
} // namespace detail

void registerHIVMDialectExtension(mlir::DialectRegistry &registry);

} // namespace hivm
} // namespace bishengir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDIALECTEXTENSION_H