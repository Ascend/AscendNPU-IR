//===- bishengir.c - Test of BiShengIR C API ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: bishengir-capi-ir-test 2>&1 | FileCheck %s
 */

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bishengir-c/RegisterEverything.h"
#include "mlir-c/BuiltinTypes.h"

int registerHFusionDialect(MlirContext ctx) {
  // CHECK-LABEL: @registerHFusionDialect
  fprintf(stderr, "@registerHFusionDialect\n");

  // CHECK: hfusion.elemwise_unary is_registered: 1
  fprintf(stderr, "hfusion.elemwise_unary is_registered: %d\n",
          mlirContextIsRegisteredOperation(
              ctx, mlirStringRefCreateFromCString("hfusion.elemwise_unary")));

  return 0;
}

int registerHIVMDialect(MlirContext ctx) {
  // CHECK-LABEL: @registerHIVMDialect
  fprintf(stderr, "@registerHIVMDialect\n");

  // hivm hir ops
  const int nOps = 25;
  char hivmOps[][20] = {
      "vabs",    "vadd",          "vand",   "vbrc",           "vcast",
      "vdiv",    "vexp",          "vln",    "vmax",           "vmin",
      "vmul",    "vnot",          "vor",    "vrec",           "vreduce",
      "vrelu",   "vsqrt",         "vsub",   "convert_layout", "copy",
      "fixpipe", "get_block_idx", "matmul", "mmadL1",         "nd2nz"};

  for (int i = 0; i < nOps; ++i) {
    char hivmOp[30] = {""};
    strcat(hivmOp, "hivm.hir.");
    strcat(hivmOp, hivmOps[i]);

    char format[60] = {""};
    strcat(format, hivmOp);
    strcat(format, " is_registered: %d\n");

    fprintf(stderr, format,
            mlirContextIsRegisteredOperation(
                ctx, mlirStringRefCreateFromCString(hivmOp)));
  }

  // CHECK: hivm.hir.vabs is_registered: 1
  // CHECK: hivm.hir.vadd is_registered: 1
  // CHECK: hivm.hir.vand is_registered: 1
  // CHECK: hivm.hir.vbrc is_registered: 1
  // CHECK: hivm.hir.vcast is_registered: 1
  // CHECK: hivm.hir.vdiv is_registered: 1
  // CHECK: hivm.hir.vexp is_registered: 1
  // CHECK: hivm.hir.vln is_registered: 1
  // CHECK: hivm.hir.vmax is_registered: 1
  // CHECK: hivm.hir.vmin is_registered: 1
  // CHECK: hivm.hir.vmul is_registered: 1
  // CHECK: hivm.hir.vnot is_registered: 1
  // CHECK: hivm.hir.vor is_registered: 1
  // CHECK: hivm.hir.vrec is_registered: 1
  // CHECK: hivm.hir.vreduce is_registered: 1
  // CHECK: hivm.hir.vrelu is_registered: 1
  // CHECK: hivm.hir.vsqrt is_registered: 1
  // CHECK: hivm.hir.vsub is_registered: 1
  // CHECK: hivm.hir.convert_layout is_registered: 1
  // CHECK: hivm.hir.copy is_registered: 1
  // CHECK: hivm.hir.fixpipe is_registered: 1
  // CHECK: hivm.hir.get_block_idx is_registered: 1
  // CHECK: hivm.hir.matmul is_registered: 1
  // CHECK: hivm.hir.mmadL1 is_registered: 1
  // CHECK: hivm.hir.nd2nz is_registered: 1

  return 0;
}

int registerAnnotationDialect(MlirContext ctx) {
  // CHECK-LABEL: @registerAnnotationDialect
  fprintf(stderr, "@registerAnnotationDialect\n");

  // CHECK: annotation.mark is_registered: 1
  fprintf(stderr, "annotation.mark is_registered: %d\n",
          mlirContextIsRegisteredOperation(
              ctx, mlirStringRefCreateFromCString("annotation.mark")));

  return 0;
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  bishengirRegisterAllDialects(ctx);

  if (registerHFusionDialect(ctx))
    return 1;

  if (registerHIVMDialect(ctx))
    return 1;

  if (registerAnnotationDialect(ctx))
    return 1;

  // CHECK: DESTROY MAIN CONTEXT
  fprintf(stderr, "DESTROY MAIN CONTEXT\n");
  mlirContextDestroy(ctx);

  return EXIT_SUCCESS;
}
