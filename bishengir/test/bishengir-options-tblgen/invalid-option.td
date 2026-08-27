// RUN: not bishengir-options-tblgen -gen-compile-configs %s -I %p/../../include/ 2>&1 | FileCheck %s --check-prefix=COMPILE-CONFIGS

include "bishengir/Tools/ConfigOptions/Options.td"

// COMPILE-CONFIGS: Unknown config name
def Foo : Option {
  let description = "Foooooo.";
  let type = "int64_t";
  let defaultValue = "1";
}
