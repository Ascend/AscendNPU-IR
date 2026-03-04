//===- pass.c - Simple test of C APIs -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
  TODO: This unittest is unsupported when we build triton because somehow the
  nvgpu dialect was registered multiple times. Fix this later.

  UNSUPPORTED: enable_triton_ir_compile
  RUN: bishengir-capi-pass-test 2>&1
 */

#include "bishengir-c/Dialect/Annotation.h"
#include "bishengir-c/Dialect/HFusion.h"
#include "bishengir-c/Dialect/HIVM.h"
#include "bishengir-c/RegisterEverything.h"
#include "mlir-c/RegisterEverything.h"

#include <stdio.h>
#include <stdlib.h>

static void registerMLIRAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

void testHIVMPasses(MlirContext ctx) {
  const char *funcAsm = //
      "func.func @test_vbrc_temp_buffer_first_axis_unalign() {   \n"
      "  %src = memref.alloc() : memref<1x3xf16>                 \n"
      "  %dst = memref.alloc() : memref<10x3xf16>                \n"
      "                                                          \n"
      "  hivm.hir.vbrc ins(%src : memref<1x3xf16>)               \n"
      "                outs(%dst : memref<10x3xf16>)             \n"
      "                broadcast_dims = [0]                      \n"
      "                                                          \n"
      "  return                                                  \n"
      "}                                                         \n";
  MlirOperation func =
      mlirOperationCreateParse(ctx, mlirStringRefCreateFromCString(funcAsm),
                               mlirStringRefCreateFromCString("funcAsm"));
  if (mlirOperationIsNull(func)) {
    fprintf(stderr, "Unexpected failure parsing asm.\n");
    exit(EXIT_FAILURE);
  }

  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirPass allocExtraBufferPass = mlirCreateHIVMAllocExtraBuffer();
    mlirPassManagerAddOwnedPass(pm, allocExtraBufferPass);
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, func);
    if (mlirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running pass manager.\n");
      exit(EXIT_FAILURE);
    }
    mlirPassManagerDestroy(pm);
  }

  mlirOperationDestroy(func);
}

void testHFusionOpFusion(MlirContext ctx) {
  const char *moduleAsm = //
      "module {                                                             \n"
      "func.func @test_graph_a(%arg0: tensor<?x?xf32>)                      \n"
      "	    -> (tensor<?x?xf32>, tensor<?x?xf32>) {                         \n"
      "	%c0 = arith.constant 0 : index                                      \n"
      "	%c1 = arith.constant 1 : index                                      \n"
      "	%0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>                        \n"
      "	%1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>                        \n"
      "	%2 = tensor.empty(%0, %1) : tensor<?x?xf32>                         \n"
      " %3 = linalg.matmul                                                  \n"
      "        ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)         \n"
      "	       outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>                \n"
      "	%4 = tensor.empty(%0, %1) : tensor<?x?xf32>                         \n"
      "	%5 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>}            \n"
      "	       ins(%3 : tensor<?x?xf32>)                                    \n"
      "	       outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>                \n"
      "	%6 = tensor.empty(%0, %1) : tensor<?x?xf32>                         \n"
      "	%7 = linalg.matmul ins(%arg0, %5 : tensor<?x?xf32>, tensor<?x?xf32>)\n"
      "		           outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>    \n"
      "	%8 = tensor.empty(%0, %1) : tensor<?x?xf32>                         \n"
      "	%9 = linalg.matmul                                                  \n"
      "        ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)         \n"
      "	       outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>                \n"
      "	%10 = tensor.empty(%0, %1) : tensor<?x?xf32>                        \n"
      "	%11 = linalg.elemwise_binary {add, fun = #linalg.binary_fn<add>}    \n"
      "		ins(%5, %9 : tensor<?x?xf32>, tensor<?x?xf32>)              \n"
      "		outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>              \n"
      "	return %11, %7 : tensor<?x?xf32>, tensor<?x?xf32>                   \n"
      "}}                                                                   \n";

  MlirOperation module =
      mlirOperationCreateParse(ctx, mlirStringRefCreateFromCString(moduleAsm),
                               mlirStringRefCreateFromCString("moduleAsm"));
  if (mlirOperationIsNull(module)) {
    fprintf(stderr, "Unexpected failure parsing asm.\n");
    exit(EXIT_FAILURE);
  }

  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirPass pass = mlirCreateHFusionHFusionOpFusion();
    mlirPassManagerAddOwnedPass(pm, pass);
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, module);
    if (mlirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running pass manager.\n");
      exit(EXIT_FAILURE);
    }
    mlirPassManagerDestroy(pm);
  }

  mlirOperationDestroy(module);
}

void testAnnotationPasses(MlirContext ctx) {
  const char *funcAsm = //
      "func.func @mark(                                            \n"
      "    %t1 : tensor<?xf32> {bufferization.writable = true},    \n"
      "    %t2 : tensor<?xf32> {bufferization.writable = true},    \n"
      "    %c : i1)                                                \n"
      "  -> (tensor<?xf32>, tensor<?xf32>)                         \n"
      "{                                                           \n"
      "  %cst = arith.constant 0.0 : f32                           \n"
      "  %idx = arith.constant 0 : index                           \n"
      "  %w = tensor.insert %cst into %t1[%idx] : tensor<?xf32>    \n"
      "  %s = arith.select %c, %t1, %t2 : tensor<?xf32>            \n"
      "  annotation.mark %s {attr = 2 : i32} : tensor<?xf32>       \n"
      "  return %s, %w : tensor<?xf32>, tensor<?xf32>              \n"
      "}                                                           \n";

  MlirOperation func =
      mlirOperationCreateParse(ctx, mlirStringRefCreateFromCString(funcAsm),
                               mlirStringRefCreateFromCString("funcAsm"));
  if (mlirOperationIsNull(func)) {
    fprintf(stderr, "Unexpected failure parsing asm.\n");
    exit(EXIT_FAILURE);
  }

  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirPass pass = mlirCreateAnnotationAnnotationLowering();
    mlirPassManagerAddOwnedPass(pm, pass);
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, func);
    if (mlirLogicalResultIsFailure(success)) {
      fprintf(stderr, "Unexpected failure running pass manager.\n");
      exit(EXIT_FAILURE);
    }
    mlirPassManagerDestroy(pm);
  }

  mlirOperationDestroy(func);
}

void testHFusionAutoSchedule(MlirContext ctx) {
  const char *moduleAsm = //
      "module {                                                              \n"
      "  func.func @test_fusing_interm_producers(%arg0: tensor<1024xf32>,    \n"
      "				                 %arg1: tensor<1024xf32>,    \n"
      "					         %arg2: tensor<1024xf32>)    \n"
      "					         -> tensor<1024xf32>         \n"
      "	attributes {hfusion.fusion_kind=#hfusion.fusion_kind<PURE_ELEMWISE>}{\n"
      "    %0 = tensor.empty() : tensor<1024xf32>                            \n"
      "    %1 = tensor.empty() : tensor<1024xf32>                            \n"
      "    %2 = tensor.empty() : tensor<1024xf32>                            \n"
      "    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul> }       \n"
      "		   ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>)    \n"
      "		   outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>           \n"
      "    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}        \n"
      "	       ins(%3, %arg2 : tensor<1024xf32>, tensor<1024xf32>)           \n"
      "		   outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>           \n"
      "    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>}        \n"
      "	       ins(%3, %4 : tensor<1024xf32>, tensor<1024xf32>)              \n"
      "		   outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>           \n"
      "    return %5 : tensor<1024xf32>                                      \n"
      "  }                                                                   \n"
      "}                                                                    \n";

  MlirOperation module =
      mlirOperationCreateParse(ctx, mlirStringRefCreateFromCString(moduleAsm),
                               mlirStringRefCreateFromCString("moduleAsm"));
  if (mlirOperationIsNull(module)) {
    fprintf(stderr, "Unexpected failure parsing asm.\n");
    exit(EXIT_FAILURE);
  }

  {
    MlirPassManager pm = mlirPassManagerCreate(ctx);
    MlirPass pass = mlirCreateHFusionAutoSchedule();
    mlirPassManagerAddOwnedPass(pm, pass);
    MlirLogicalResult success = mlirPassManagerRunOnOp(pm, module);
    if (mlirLogicalResultIsFailure(success))
      exit(2);
    mlirPassManagerDestroy(pm);
  }

  mlirOperationDestroy(module);
}

int main(void) {
  // register bishengir and mlir
  MlirContext ctx = mlirContextCreate();
  bishengirRegisterAllDialects(ctx);
  bishengirRegisterAllPasses();
  registerMLIRAllUpstreamDialects(ctx);
  mlirRegisterAllPasses();

  // test
  testHIVMPasses(ctx);
  testHFusionOpFusion(ctx);
  testHFusionAutoSchedule(ctx);
  testAnnotationPasses(ctx);

  // destroy ctx
  mlirContextDestroy(ctx);
  return 0;
}
