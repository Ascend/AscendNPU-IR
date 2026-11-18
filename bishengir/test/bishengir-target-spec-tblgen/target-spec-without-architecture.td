// RUN: bishengir-target-spec-tblgen -gen-target-spec-decls %s \
// RUN:   | FileCheck %s --check-prefix=DECL \
// RUN:     --implicit-check-not=NPUArchitecture \
// RUN:     --implicit-check-not=getArchitecture --implicit-check-not=getArch
// RUN: bishengir-target-spec-tblgen -gen-target-spec-defs %s \
// RUN:   | FileCheck %s --check-prefix=DEF \
// RUN:     --implicit-check-not=NPUArchitecture \
// RUN:     --implicit-check-not=getArchitecture --implicit-check-not=getArch

// Generic schemas remain usable without opting into the NPU Arch field.
class TargetSpec<string name> {
  string Name = name;
  int IntSpec = 17;
}
def GenericSpec : TargetSpec<"Generic">;

// DECL: enum class TargetDevice {
// DECL: Generic,
// DECL: struct TargetSpec {
// DECL: int IntSpec;
// DECL: Attribute getSpecEntry(DeviceSpec spec, OpBuilder& builder) const;

// DEF: static const TargetSpec specs[] = {
// DEF: TargetDevice::Generic,
// DEF-NEXT: 17,
// DEF: if (specEntry == DeviceSpec::INT_SPEC)
// DEF: return wrap(IntSpec, builder);
