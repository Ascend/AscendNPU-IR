//===- HACCToLLVM.h - HACC to LLVM ------------------------------*- C++ -*-===//
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
// Define conversions from the HACC dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H
#define BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H

#include <memory>
#include <string>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTHACCTOLLVM
#include "bishengir/Conversion/Passes.h.inc"

namespace hacc {
/// Collect the patterns to convert from the HACC dialect to LLVM.
void populateHACCToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          const std::string &deviceFilePath);
} // namespace hacc

/// Creates a pass to convert the HACC dialect into the LLVMIR dialect.
std::unique_ptr<Pass> createConvertHACCToLLVMPass();

std::unique_ptr<Pass>
createConvertHACCToLLVMPass(const ConvertHACCToLLVMOptions &option);

} // namespace mlir

#endif // BISHENGIR_CONVERSION_HACCTOLLVM_HACCTOLLVM_H
