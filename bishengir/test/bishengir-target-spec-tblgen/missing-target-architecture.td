// RUN: not bishengir-target-spec-tblgen -I %bishengir_src_root/include \
// RUN:   -gen-target-spec-decls %s 2>&1 | FileCheck %s
// RUN: not bishengir-target-spec-tblgen -I %bishengir_src_root/include \
// RUN:   -gen-target-spec-defs %s 2>&1 | FileCheck %s

include "bishengir/Target/NPU/NPUArchitecture.td"

class TargetSpec<string name> {
  string Name = name;
  NPUArchitecture Arch = ?;
}

// CHECK: error: TargetSpec 'MissingArchitecture'
// CHECK-SAME: must define required field 'Arch'
def MissingArchitecture : TargetSpec<"MissingArchitecture">;
