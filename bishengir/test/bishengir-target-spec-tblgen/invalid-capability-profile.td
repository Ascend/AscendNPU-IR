// clang-format off
// RUN: not bishengir-target-spec-tblgen -DDUP_PROFILE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=PROFILE
// RUN: not bishengir-target-spec-tblgen -DDUP_COVERAGE -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=COVERAGE
// RUN: not bishengir-target-spec-tblgen -DWRONG_COVERAGE -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=WRONG
// RUN: not bishengir-target-spec-tblgen -DDUP_ROW -gen-target-capability-defs %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ROW
// RUN: not bishengir-target-spec-tblgen -DDUP_ALIAS_ROW -gen-target-capability-decls %s \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include -I %llvm_src_root/include 2>&1 | FileCheck %s --check-prefix=ROW
// clang-format on

include "bishengir/Target/NPU/NPUCapabilityProfileBase.td"

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}
def ModeA : I32EnumAttrCase<"A", 0, "a">;
def ModeB : I32EnumAttrCase<"B", 1, "b">;
def ModeEnum : I32EnumAttr<"Mode", "test modes", [ModeA, ModeB]>;
def ModeAttr : EnumAttr<TestDialect, ModeEnum, "mode">;
def Mode : EnumAttrCapabilityParameter<ModeAttr>;
def Enabled : BoolAttrCapabilityParameter<>;
def ElementType : CapabilitySlotProperty<MLIRTypeDomain>;
def StorageType : CapabilitySlotProperty<MLIRTypeDomain>;
def Src : CapabilitySlot<[ElementType, StorageType]>;
def SrcType : SlotField<Src, ElementType>;
def SrcStorage : SlotField<Src, StorageType>;
def Schema : RequestSchema<[Src], [Mode, Enabled]>;
def Key : CapabilityKey<Schema>;
def Row : SupportCase<Key, [TypeSignature<[SrcType, SrcStorage], [F16, F32]>,
                            TypeSignature<[SrcType, SrcStorage], [F32, F32]>,
                            EnumAnyOf<Mode, [ModeA, ModeB]>,
                            BoolValueConstraint<Enabled, 1>]>;
def Coverage : ArchitectureCapabilityCoverage<Key, [Row], CompleteProfile>;
def Profile : ArchitectureCapabilityProfile<Dav_C220, [Coverage]>;

#ifdef DUP_PROFILE
def OtherProfile : ArchitectureCapabilityProfile<Dav_C220, []>;
#endif

#ifdef DUP_ALIAS_ROW
def F16Alias : TypeAlias<F16>;
def AliasedRow
    : SupportCase<Key, [BoolValueConstraint<Enabled, 1>,
                        EnumAnyOf<Mode, [ModeB, ModeA]>,
                        TypeSignature<[SrcStorage, SrcType], [F32, F32]>,
                        TypeSignature<[SrcStorage, SrcType], [F32, F16Alias]>]>;
def OtherCoverage
    : ArchitectureCapabilityCoverage<Key, [Row, AliasedRow], CompleteProfile>;
#endif
#ifdef DUP_COVERAGE
def OtherCoverage : ArchitectureCapabilityCoverage<Key, [], PartialProfile>;
def OtherProfile
    : ArchitectureCapabilityProfile<Dav_C310, [Coverage, OtherCoverage]>;
#endif
#ifdef WRONG_COVERAGE
def OtherKey : CapabilityKey<Schema>;
def OtherCoverage
    : ArchitectureCapabilityCoverage<OtherKey, [Row], CompleteProfile>;
#endif
#ifdef DUP_ROW
// Record identity, tuple field order, and row order do not make a new fact.
def ReorderedRow
    : SupportCase<Key, [BoolValueConstraint<Enabled, 1>,
                        EnumAnyOf<Mode, [ModeB, ModeA]>,
                        TypeSignature<[SrcStorage, SrcType], [F32, F32]>,
                        TypeSignature<[SrcStorage, SrcType], [F32, F16]>]>;
def OtherCoverage
    : ArchitectureCapabilityCoverage<Key, [Row, ReorderedRow], CompleteProfile>;
#endif

// clang-format off
// PROFILE: error: duplicate capability profile
// COVERAGE: error: duplicate capability coverage
// WRONG: error: SupportCase capability does not match coverage
// ROW: error: duplicate normalized SupportCase
// clang-format on
