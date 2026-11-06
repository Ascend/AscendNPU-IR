// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-binding %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include | FileCheck %s
// clang-format on

include "bishengir/Dialect/HIVM/IR/Capability/Sync/SyncCapabilityBinding.td"

// The sync catalogue gets its own guard and namespace so it links beside
// fixpipe, which stays in bishengir::npu.
// CHECK: #ifndef BISHENGIR_TARGET_NPU_SYNC_CAPABILITY_BINDING_H
// CHECK: namespace bishengir {
// CHECK: namespace npu {
// CHECK: namespace sync {

// Field-free schemas: create() takes no arguments and from() cannot fail.
// CHECK: class GetBufCapabilityRequestBuilder
// CHECK: from(::mlir::hivm::GetBufOp op)
// CHECK: create()
// CHECK: storage.capability = CapabilityKind::GetBuf
// CHECK: storage.values.reserve(0)
// CHECK: class RlsBufCapabilityRequestBuilder
// CHECK: from(::mlir::hivm::RlsBufOp op)
// CHECK: create()
// CHECK: storage.capability = CapabilityKind::RlsBuf
// CHECK: storage.values.reserve(0)
// try_wait carries one enum field bound to the sync_mode attribute.
// CHECK: class TryWaitCapabilityRequestBuilder
// CHECK: from(::mlir::hivm::TryWaitOp op)
// CHECK: create(::mlir::hivm::TryWaitSyncMode syncMode)
// CHECK: getAttr("sync_mode")
// CHECK: dyn_cast<::mlir::hivm::TryWaitSyncModeAttr>
// A missing or mistyped sync_mode attribute is an illegal binding input.
// CHECK: return ::std::nullopt
// CHECK: storage.capability = CapabilityKind::TryWait
// CHECK: storage.values.reserve(1)

// The catalogue traits drive the templated verify-capability pass. Every
// call is qualified with the scope namespace: an unqualified call that
// takes an NPUArchitectureKind would pick up the fixpipe overload in
// bishengir::npu through ADL once both binding headers share one
// translation unit.
// CHECK: struct CapabilityCatalogue {
// CHECK: ::bishengir::npu::sync::visitBoundCapabilityOps
// CHECK: ::bishengir::npu::sync::queryNativeSupport
// CHECK: ::bishengir::npu::sync::lookupArchitectureCapabilityProfile
// CHECK: } // namespace sync
// CHECK: } // namespace npu
// CHECK: } // namespace bishengir
