#ifndef BISHENGIR_CONVERSION_PROTONTOHIVM_PROTONTOHIVM_H
#define BISHENGIR_CONVERSION_PROTONTOHIVM_PROTONTOHIVM_H

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace bishengir::triton::proton::a5 {

#define GEN_PASS_DECL_CONVERTPROTONTOHIVM
#define GEN_PASS_DECL_GETPROTONMETADATA
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createConvertProtonToHIVMPass();
std::unique_ptr<mlir::Pass>
createGetProtonMetadataPass(const GetProtonMetadataOptions &options = {});

} // namespace bishengir::triton::proton::a5

#endif // BISHENGIR_CONVERSION_PROTONTOHIVM_PROTONTOHIVM_H
