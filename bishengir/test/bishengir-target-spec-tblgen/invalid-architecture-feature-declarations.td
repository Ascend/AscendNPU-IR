// RUN: split-file %s %t
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/empty-feature.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-FEATURE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/empty-value.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-VALUE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/empty-enum.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-ENUM
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/reserved-value.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RESERVED-VALUE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/reserved-spelling.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RESERVED-SPELLING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/duplicate-value.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-VALUE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls \
// RUN:   %t/duplicate-value-spelling.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-VALUE-SPELLING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/duplicate-feature.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-FEATURE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/duplicate-accessor.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-ACCESSOR
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/reserved-field.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RESERVED-FIELD
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/identifier-collision.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=IDENTIFIER-COLLISION
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/unsupported-setting.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNSUPPORTED-SETTING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls \
// RUN:   %t/duplicate-feature-spelling.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-FEATURE-SPELLING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/keyword-type.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=KEYWORD-TYPE
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/keyword-accessor.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=KEYWORD-ACCESSOR
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include \
// RUN:   -gen-npu-architecture-feature-decls %t/keyword-field.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=KEYWORD-FIELD

// EMPTY-FEATURE: architecture feature C++ name must not be empty
// EMPTY-VALUE: architecture feature value C++ name must not be empty
// EMPTY-ENUM: architecture enum feature has no values: empty-enum
// RESERVED-VALUE: architecture feature value C++ name 'Unknown'
// RESERVED-VALUE-SAME: is reserved for kUnknown
// RESERVED-SPELLING: architecture feature value spelling 'unknown' is reserved
// DUP-VALUE: duplicate C++ value for architecture feature 'mode': Enabled
// DUP-VALUE-SPELLING: duplicate value spelling for architecture feature 'mode':
// DUP-VALUE-SPELLING-SAME: shared
// DUP-FEATURE: duplicate architecture feature C++ name: Mode
// DUP-ACCESSOR: duplicate architecture feature accessor: queryMode
// RESERVED-FIELD: architecture feature profile field 'architecture' is reserved
// IDENTIFIER-COLLISION: duplicate generated architecture feature
// IDENTIFIER-COLLISION-SAME: C++ identifier: stringifyMode
// UNSUPPORTED-SETTING: unsupported architecture feature setting kind
// DUP-FEATURE-SPELLING: duplicate architecture feature spelling: shared
// KEYWORD-TYPE: invalid architecture feature C++ name: class
// KEYWORD-ACCESSOR: invalid architecture feature accessor: switch
// KEYWORD-FIELD: invalid architecture feature profile field: class

//--- empty-feature.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Empty : EnumArchitectureFeature<"">;
def Value : EnumArchitectureFeatureValue<Empty, "Enabled">;

//--- empty-value.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def Empty : EnumArchitectureFeatureValue<Mode, "">;

//--- empty-enum.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Empty : EnumArchitectureFeature<"EmptyEnum">;

//--- reserved-value.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def Reserved : EnumArchitectureFeatureValue<Mode, "Unknown", "explicit">;

//--- reserved-spelling.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def Reserved : EnumArchitectureFeatureValue<Mode, "Enabled", "unknown">;

//--- duplicate-value.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def First : EnumArchitectureFeatureValue<Mode, "Enabled", "first">;
def Second : EnumArchitectureFeatureValue<Mode, "Enabled", "second">;

//--- duplicate-value-spelling.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def First : EnumArchitectureFeatureValue<Mode, "First", "shared">;
def Second : EnumArchitectureFeatureValue<Mode, "Second", "shared">;

//--- duplicate-feature.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def FirstMode : EnumArchitectureFeature<"Mode", "first-mode", "queryFirst">;
def First : EnumArchitectureFeatureValue<FirstMode, "Enabled">;
def SecondMode : EnumArchitectureFeature<"Mode", "second-mode", "querySecond">;
def Second : EnumArchitectureFeatureValue<SecondMode, "Enabled">;

//--- duplicate-accessor.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def FirstMode : EnumArchitectureFeature<"FirstMode", "", "queryMode">;
def First : EnumArchitectureFeatureValue<FirstMode, "Enabled">;
def SecondMode : EnumArchitectureFeature<"SecondMode", "", "queryMode">;
def Second : EnumArchitectureFeatureValue<SecondMode, "Enabled">;

//--- reserved-field.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Architecture : EnumArchitectureFeature<"Architecture">;
def Value : EnumArchitectureFeatureValue<Architecture, "Enabled">;

//--- identifier-collision.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode", "", "stringifyMode">;
def Value : EnumArchitectureFeatureValue<Mode, "Enabled">;

//--- unsupported-setting.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode">;
def Value : EnumArchitectureFeatureValue<Mode, "Enabled">;
def Custom : ArchitectureFeatureSetting<Mode>;

//--- duplicate-feature-spelling.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def FirstMode : EnumArchitectureFeature<"FirstMode", "shared">;
def First : EnumArchitectureFeatureValue<FirstMode, "Enabled">;
def SecondMode : EnumArchitectureFeature<"SecondMode", "shared">;
def Second : EnumArchitectureFeatureValue<SecondMode, "Enabled">;

//--- keyword-type.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"class">;
def Value : EnumArchitectureFeatureValue<Mode, "Enabled">;

//--- keyword-accessor.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Mode", "", "switch">;
def Value : EnumArchitectureFeatureValue<Mode, "Enabled">;

//--- keyword-field.td
include "bishengir/Target/NPU/NPUArchitectureFeatureBase.td"
def Mode : EnumArchitectureFeature<"Class">;
def Value : EnumArchitectureFeatureValue<Mode, "Enabled">;
