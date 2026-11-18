// RUN: split-file %s %t
// RUN: not bishengir-target-spec-tblgen -gen-target-spec-decls \
// RUN:   %t/wrong-record.td 2>&1 | FileCheck %s --check-prefix=RECORD
// RUN: not bishengir-target-spec-tblgen -gen-target-spec-defs \
// RUN:   %t/wrong-record.td 2>&1 | FileCheck %s --check-prefix=RECORD
// RUN: not bishengir-target-spec-tblgen -gen-target-spec-decls \
// RUN:   %t/string-value.td 2>&1 | FileCheck %s --check-prefix=STRING
// RUN: not bishengir-target-spec-tblgen -gen-target-spec-defs \
// RUN:   %t/string-value.td 2>&1 | FileCheck %s --check-prefix=STRING

// RECORD: error: TargetSpec 'WrongArchitecture' field 'Arch'
// RECORD-SAME: must reference an NPUArchitecture
// STRING: error: TargetSpec 'StringArchitecture' field 'Arch'
// STRING-SAME: must reference an NPUArchitecture

//--- wrong-record.td
// A record with a Name is not sufficient: Arch must derive from
// NPUArchitecture.
class OtherRecord<string name> { string Name = name; }
def NotAnArchitecture : OtherRecord<"dav-c220">;
class TargetSpec<string name> {
  string Name = name;
  OtherRecord Arch = NotAnArchitecture;
}
def WrongArchitecture : TargetSpec<"WrongArchitecture">;

//--- string-value.td
class TargetSpec<string name> {
  string Name = name;
  string Arch = "dav-c220";
}
def StringArchitecture : TargetSpec<"StringArchitecture">;
