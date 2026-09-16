//===- Transforms.h - HIVM Dialect Transformation Entrypoints ---*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMS_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

/// Collect a set of patterns to lower HIVM ops to ops that map to LLVM
/// intrinsics.
void populateHIVMLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Configure the target to support lowering HIVM ops to ops that map to LLVM
/// intrinsics.
void configureHIVMLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_H
