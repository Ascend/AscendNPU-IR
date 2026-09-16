// RUN: not bishengir-target-spec-tblgen -gen-target-spec-decls %s 2>&1 | FileCheck %s --check-prefix=DECL
// RUN: not bishengir-target-spec-tblgen -gen-target-spec-defs %s 2>&1 | FileCheck %s --check-prefix=DEF

class TargetSpec <string name> {
  string Name = name;
  // DEF: Unsupported spec type: dag
  // DECL: Unsupported spec type to map to c++ type: dag
  dag dagSpec = ?;
}

def X1;

def Spec : TargetSpec<"Unsupported"> {
  let dagSpec = (X1 "Test");
}
