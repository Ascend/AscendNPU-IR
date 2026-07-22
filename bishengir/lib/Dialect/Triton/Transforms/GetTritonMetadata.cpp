//===- GetTritonMetadata.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that dumps triton metadata to a file or stdout.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Triton/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/JSON.h"

#include "proton/Dialect/include/Analysis/ScopeIdAllocation.h"

#include <fstream>

namespace bishengir {
namespace triton {

#define GEN_PASS_DEF_GETTRITONMETADATA
#include "bishengir/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;

class GetTritonMetadataPass
    : public impl::GetTritonMetadataBase<GetTritonMetadataPass> {
public:
  explicit GetTritonMetadataPass(
      const GetTritonMetadataOptions &options)
      : impl::GetTritonMetadataBase<GetTritonMetadataPass>(
            options) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::json::Object metadata;

    // Helper: returns integer or null if attr not present
    auto getIntAttrOrNull = [](mlir::ModuleOp module, llvm::StringRef name) -> std::optional<int64_t> {
      if (auto attr = module->getAttrOfType<mlir::IntegerAttr>(name)) {
        return attr.getInt();
      }
      return std::nullopt;
    };

    // TODO change this to ttg.total-num-warps when ascend has Warp Specialize
    if (auto val = getIntAttrOrNull(module, "ttg.num-warps"))
      metadata["num_warps"] = *val;
    else {
      module.emitError("module missing 'ttg.num-warps' attribute");
      signalPassFailure();
      return;
    }

    // shared
    metadata["shared"] = getIntAttrOrNull(module, "ttg.shared").value_or(0);
    metadata["global_scratch_size"] = getIntAttrOrNull(module, "ttg.global_scratch_memory_size").value_or(0);
    metadata["global_scratch_align"] = getIntAttrOrNull(module, "ttg.global_scratch_memory_alignment").value_or(0);

    // profile scratch defaults
    metadata["profile_scratch_size"] = getIntAttrOrNull(module, "ttg.profile_scratch_memory_size").value_or(0);
    metadata["profile_scratch_align"] = getIntAttrOrNull(module, "ttg.profile_scratch_memory_alignment").value_or(1);

    llvm::json::Value json(std::move(metadata));

    if (tritonMetadataOutput == "") {
      // do nothing
    }
    else if (tritonMetadataOutput == "--") {
      // Print to stdout
      llvm::outs() << json << "\n";
    } else {
      // Write to file
      std::error_code ec;
      llvm::raw_fd_ostream os(tritonMetadataOutput, ec);
      if (ec) {
        llvm::errs() << "Error: Failed to open file "
                     << tritonMetadataOutput 
                     << ": " << ec.message() << "\n";
        return;
      }
      os << json;
      os.close();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createGetTritonMetadataPass(
    const GetTritonMetadataOptions &options) {
  return std::make_unique<GetTritonMetadataPass>(options);
}

} // namespace triton
} // namespace bishengir