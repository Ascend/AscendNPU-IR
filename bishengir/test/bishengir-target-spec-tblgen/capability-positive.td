// clang-format off
// RUN: bishengir-target-spec-tblgen -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include \
// RUN:   -I %bishengir_src_root/unittests/Target/Capability \
// RUN:   -gen-target-capability-decls %s | FileCheck %s --check-prefix=DECL \
// RUN:   --implicit-check-not=Optional --implicit-check-not=InputType \
// RUN:   --implicit-check-not=OutputType --implicit-check-not=CapabilityBinding \
// RUN:   --implicit-check-not=ElementTypeKind --implicit-check-not=CapabilityTypeDomain \
// RUN:   --implicit-check-not=symbolizeElementType --implicit-check-not=stringifyElementType \
// RUN:   --implicit-check-not=SlotType --implicit-check-not=SlotValue \
// RUN:   --implicit-check-not=TypeSignatureDesc --implicit-check-not=CapabilityFieldValueKind
// RUN: bishengir-target-spec-tblgen -I %bishengir_src_root/include \
// RUN:   -I %mlir_src_root/include -I %llvm_src_root/include \
// RUN:   -I %bishengir_src_root/unittests/Target/Capability \
// RUN:   -gen-target-capability-defs %s | FileCheck %s --check-prefix=DEF \
// RUN:   --implicit-check-not=HIVM --implicit-check-not=TargetEnv \
// RUN:   --implicit-check-not=TargetCapability.h --implicit-check-not=ElementTypeKind \
// RUN:   --implicit-check-not=CapabilityTypeDomain --implicit-check-not=symbolizeElementType \
// RUN:   --implicit-check-not=SlotType --implicit-check-not=SlotValue \
// RUN:   --implicit-check-not=request.signature --implicit-check-not=request.types
// clang-format on

// The very same catalogue is also generated, compiled, and executed by
// BiShengIRTargetCapabilityTests. No production dialect catalogue is included.
include "CapabilityTest.td"

// clang-format off
// DECL: namespace bishengir {
// DECL: namespace npu {
// DECL: enum class CapabilityValueDomainKind {
// DECL: MLIRTypeDomain,
// DECL: enum class CapabilitySlotPropertyKind {
// DECL: ElementType,
// DECL: enum class CapabilityKind {
// DECL: Triad,
// DECL: enum class CapabilityParameterKind {
// DECL: enum class CapabilityRequestFieldKind { Slot, Parameter };
// DECL: struct CapabilityRequestSchemaFieldDesc {
// DECL-NEXT: CapabilityRequestFieldDesc field;
// DECL-NEXT: CapabilityValueDomainKind domain;
// DECL-NEXT: };
// DECL: struct CapabilityFieldValue {
// DECL-NEXT: CapabilityRequestFieldDesc field;
// DECL-NEXT: std::variant<::mlir::Type, CapabilityValue> value;
// DECL-NEXT: };
// DECL: struct CapabilityRequest {
// DECL-NEXT: CapabilityKind capability;
// DECL-NEXT: ::llvm::ArrayRef<CapabilityFieldValue> values;
// DECL-NEXT: };
// DECL: struct TypeSignatureConstraintDesc {
// DECL-NEXT: bool (*matches)(const CapabilityRequest &) = nullptr;
// DECL-NEXT: };
// DECL: enum class NativeSupport {
// DECL-NEXT: Native,
// DECL-NEXT: Unsupported,
// DECL-NEXT: Unknown,
// DECL-NEXT: InvalidRequest,
// DECL: struct CapabilityRequestStorage {
// DECL: ::llvm::SmallVector<CapabilityFieldValue, 4> values;
// DECL: return {capability, values};
// DECL: bool isCapabilityRequestWellFormed(const CapabilityRequest &request);
// DECL: NativeSupport queryNativeSupport(

// DEF: #include "llvm/Support/ErrorHandling.h"
// DEF: static bool matchCapabilityRecord{{[0-9]+}}Signature0(const CapabilityRequest &request)
// DEF: const auto *field0 = findRequestField(request, {CapabilityRequestFieldKind::Slot,
// DEF: if (!field0)
// DEF-NEXT: ::llvm::report_fatal_error("type signature references a missing request field");
// DEF: const auto *type0 = std::get_if<::mlir::Type>(&field0->value);
// DEF: if (!type0 || !*type0)
// DEF-NEXT: ::llvm::report_fatal_error("type signature requires a non-null MLIR type");
// DEF: static const TypeSignatureConstraintDesc {{kCapabilityRecord[0-9]+}}Signatures[] =
// DEF: static const SupportCaseDesc {{kCapabilityRecord[0-9]+}}Cases[] = {k[[WILDCARD:CapabilityRecord[0-9]+]], k[[RESTRICTED:CapabilityRecord[0-9]+]]};
// DEF: static const SupportCaseDesc {{kCapabilityRecord[0-9]+}}Cases[] = {k[[RESTRICTED]], k[[WILDCARD]]};
// DEF: bool isCapabilityRequestWellFormed(const CapabilityRequest &request)
// DEF: if (supportCase.capability != request.capability)
// DEF-NEXT: ::llvm::report_fatal_error("support case and request capability mismatch");
// DEF: if (!sig.matches)
// DEF: "support case has a null type-signature matcher"
// DEF: return sig.matches(request);
// DEF: "value constraint references a missing request field"
// DEF: "value constraint requires an enum or boolean value"
// DEF: "value constraint has no allowed values"
// DEF: if (!llvm::is_contained(constraint.allowedValues, value->rawValue))
// DEF-NEXT: return SupportCaseMatchResult::Mismatch;
// DEF: NativeSupport queryNativeSupport(
// clang-format on
