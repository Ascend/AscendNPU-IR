//===- Transforms.h - HIVM Dialect Transformation Entrypoints ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
                                               RewritePatternSet &patterns,
                                               bool isRegBased = false);

/// Configure the target to support lowering HIVM ops to ops that map to LLVM
/// intrinsics.
void configureHIVMLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_TRANSFORMS_H
