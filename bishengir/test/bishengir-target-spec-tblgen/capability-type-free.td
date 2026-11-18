// clang-format off
// RUN: bishengir-target-spec-tblgen -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/unittests/Target/Capability \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include | FileCheck %s --check-prefix=DECL
// RUN: bishengir-target-spec-tblgen -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/unittests/Target/Capability \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include | FileCheck %s --check-prefix=DEF \
// RUN:   --implicit-check-not="static bool match" --implicit-check-not=SlotType
// clang-format on

// The same type-free catalogue is compiled and executed by unit tests.
include "TypeFreeCapabilityTest.td"

// clang-format off
// DECL: enum class CapabilityKind {
// DECL-NEXT: Empty,
// DECL-NEXT: ParameterOnly,
// DEF: static const SupportCaseDesc {{kCapabilityRecord[0-9]+}} = {CapabilityKind::Empty, {}, {}};
// DEF: static const SupportCaseDesc {{kCapabilityRecord[0-9]+}} = {CapabilityKind::ParameterOnly, {}, {}};
// DEF: bool isCapabilityRequestWellFormed(const CapabilityRequest &request)
// clang-format on
