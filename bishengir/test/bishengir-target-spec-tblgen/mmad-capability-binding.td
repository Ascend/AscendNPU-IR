// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-mmad-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include | FileCheck %s
// clang-format on

include "bishengir/Dialect/HIVM/IR/Capability/Cube/Mmad/MmadCapabilityBinding.td"

// CHECK: class BatchMmadCapabilityRequestBuilder
// CHECK: class MmadCapabilityRequestBuilder
// CHECK: static MmadCapabilityRequestBuilder create(
// Optional bias is filled with f32 / L1 when the operand is absent.
// CHECK: ::mlir::Float32Type::get(op->getContext())
// CHECK: ::mlir::hivm::AddressSpace::L1
// CHECK: class MmadMxCapabilityRequestBuilder
// K alignment is a divisor check, not a stored attribute.
// CHECK: % 64
// CHECK: visitBoundCapabilityOps
// CHECK: struct CapabilityCatalogue
