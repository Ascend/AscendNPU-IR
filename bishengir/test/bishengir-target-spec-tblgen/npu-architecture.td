// RUN: bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include \
// RUN:   -gen-npu-architecture-decls %s | FileCheck %s

include "bishengir/Target/NPU/NPUArchitecture.td"

// CHECK: namespace bishengir {
// CHECK-NEXT: namespace npu {
// CHECK: enum class NPUArchitectureKind {
// CHECK-NEXT: kDav920R1,
// CHECK-NEXT: kDavC220,
// CHECK-NEXT: kDavC310,
// CHECK-NEXT: kDavM300,
// CHECK-NEXT: kUnknown,
// CHECK-NEXT: };
// CHECK: inline ::llvm::StringRef
// CHECK-NEXT: stringifyNPUArchitecture(NPUArchitectureKind architecture) {
// CHECK: case NPUArchitectureKind::kDav920R1: return "dav-920r1";
// CHECK-NEXT: case NPUArchitectureKind::kDavC220: return "dav-c220";
// CHECK-NEXT: case NPUArchitectureKind::kDavC310: return "dav-c310";
// CHECK-NEXT: case NPUArchitectureKind::kDavM300: return "dav-m300";
// CHECK-NEXT: case NPUArchitectureKind::kUnknown: return "unknown";
// CHECK: inline NPUArchitectureKind
// CHECK-NEXT: symbolizeNPUArchitecture(::llvm::StringRef name) {
// CHECK: .Case("dav-920r1", NPUArchitectureKind::kDav920R1)
// CHECK-NEXT: .Case("dav-c220", NPUArchitectureKind::kDavC220)
// CHECK-NEXT: .Case("dav-c310", NPUArchitectureKind::kDavC310)
// CHECK-NEXT: .Case("dav-m300", NPUArchitectureKind::kDavM300)
// CHECK-NEXT: .Default(NPUArchitectureKind::kUnknown);
// CHECK: } // namespace npu
// CHECK-NEXT: } // namespace bishengir
