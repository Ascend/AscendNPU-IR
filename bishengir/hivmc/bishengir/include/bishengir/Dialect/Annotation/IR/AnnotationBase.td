//===- AnnotationBase.td - Base definitions for annotation dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD

include "mlir/IR/OpBase.td"

def Annotation_Dialect : Dialect {
  let name = "annotation";
  let description = [{
    Annotation dialects for mark operations to
    define some extra attrs for a certain op:

    ```mlir
    // Mark extra attrs
    annotation.mark %a { attr-dict } : f64
    ```
  }];
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::annotation";
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
