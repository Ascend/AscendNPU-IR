//===-------- HIVMCMain.cpp - HIVMC Compile Tool Support C++-*--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Triton/Pipelines/Passes.h"
#include "bishengir/Pass/PassManager.h"
#include "bishengir/Tools/hivmc/AdapterSanitizer.h"
#include "bishengir/Tools/hivmc/Config.h"
#include "bishengir/Tools/hivmc/HIVMC.h"
#include "bishengir/Tools/hivmc/HIVMCA3.h"
#include "bishengir/Tools/hivmc/HIVMCA5.h"

#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"

#include "bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.h"

#define DEBUG_TYPE "bishengir-compile"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

using namespace bishengir;
using namespace llvm;
using namespace mlir;
using namespace object;

LogicalResult bishengir::runHIVMCCompile(mlir::ModuleOp module, HIVMCMainConfig config) {
  if (failed(checkOptionValidity(config))) {
    return failure();
  }
  CompileTiming timing;

    LogicalResult runPipelineStatus = failure();
    if (config.shouldCompileA3()) {
      runPipelineStatus = runHIVMCCompileA3(module, config);
    } else if (config.shouldCompileA5()) {
      runPipelineStatus = runHIVMCCompileA5(module, config);
    }

    return runPipelineStatus;
}
