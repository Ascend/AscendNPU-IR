// RUN: bishengir-target-spec-tblgen -gen-target-spec-decls %s | FileCheck %s --check-prefix=DECL
// RUN: bishengir-target-spec-tblgen -gen-target-spec-defs %s | FileCheck %s --check-prefix=DEF

class TargetSpec <string name> {
  string Name = name;
  // Currently, spec entries must have string or int type
  string GroupName = ?;
  int IntSpec = ?;
}

class Group1Spec <string name> : TargetSpec<name> {
  let GroupName = "Group1";
}

class Group2Spec <string name> : TargetSpec<name> {
  let GroupName = "Group2";
}

def Group1_1_Spec : Group1Spec<"1_1"> {
  let IntSpec = 0x11;
}

def Group2_1_Spec : Group2Spec<"2_1"> {
  let IntSpec = 0x21;
}

//============= Check declarations =================//

// DECL: #ifndef BISHENGIR_TARGET_SPEC_H
// DECL: #define BISHENGIR_TARGET_SPEC_H

// DECL: namespace mlir {
// DECL: namespace hacc {
// DECL: enum class TargetDevice {
// DECL:   1_1,
// DECL:   2_1,
// DECL:   Unknown
// DECL: };

// DECL: TargetDevice symbolizeTargetDeviceEnum(::llvm::StringRef);

// DECL: struct TargetSpec {
// DECL:   TargetDevice device;
// DECL:   string GroupName;
// DECL:   int IntSpec;

// DECL: public:
// DECL:   Attribute getSpecEntry(DeviceSpec spec, OpBuilder& builder) const;
// DECL: };

// DECL: } // namespace hacc
// DECL: } // namespace mlir

// DECL: #endif // BISHENGIR_TARGET_SPEC_H

//============= Check definitions ==================//

// DEF: #include "NPUTargetSpec.h.inc"

// DEF: namespace mlir {
// DEF: namespace hacc {
// DEF: static const TargetSpec specs[] = {
// DEF:  {
// DEF:    TargetDevice::1_1,
// DEF:    "Group1",
// DEF:    17,
// DEF:  },
// DEF:  {
// DEF:    TargetDevice::2_1,
// DEF:    "Group2",
// DEF:    33,
// DEF:   },
// DEF: };

// DEF: Attribute TargetSpec::getSpecEntry(DeviceSpec specEntry, OpBuilder& builder) const {
// DEF:   if (specEntry == DeviceSpec::GROUP_NAME) {
// DEF:     return wrap(GroupName, builder);
// DEF:   }
// DEF:   if (specEntry == DeviceSpec::INT_SPEC) {
// DEF:     return wrap(IntSpec, builder);
// DEF:   }
// DEF:   return Attribute();
// DEF: }

// DEF: TargetDevice symbolizeTargetDeviceEnum(::llvm::StringRef str){
// DEF:   return ::llvm::StringSwitch<TargetDevice>(str)
// DEF:       .Case("1_1", TargetDevice::1_1)
// DEF:       .Case("2_1", TargetDevice::2_1)
// DEF:       .Default(TargetDevice::Unknown);
// DEF: }

// DEF: } // namespace hacc
// DEF: } // namespace mlir
