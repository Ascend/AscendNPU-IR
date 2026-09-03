//===-- Passes.td - Annotation pass definition file------------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_PASSES
#define BISHENGIR_DIALECT_ANNOTATION_PASSES

include "mlir/Pass/PassBase.td"

def AnnotationLowering : Pass<"annotation-lowering"> {
  let summary = "Lowering the annotation dialect.";
  let constructor = "mlir::annotation::createAnnotationLoweringPass()";
}

#endif // BISHENGIR_DIALECT_ANNOTATION_PASSES
