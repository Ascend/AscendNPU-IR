// RUN: split-file %s %t
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/duplicate-spelling.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DUP-SPELLING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/duplicate-enumerator.td 2>&1 | FileCheck %s --check-prefix=DUP-ENUM
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/empty-spelling.td 2>&1 | FileCheck %s --check-prefix=EMPTY-SPELLING
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/empty-enumerator.td 2>&1 | FileCheck %s --check-prefix=EMPTY-ENUM
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/invalid-enumerator.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVALID-ENUM
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/reserved-enumerator.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RESERVED-ENUM
// RUN: not bishengir-target-spec-tblgen \
// RUN:   -I %bishengir_src_root/include -I %mlir_src_root/include \
// RUN:   -I %llvm_src_root/include -gen-npu-architecture-decls \
// RUN:   %t/reserved-spelling.td 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RESERVED-SPELLING

// DUP-SPELLING: error: duplicate NPU architecture spelling: dav-c220
// DUP-ENUM: error: duplicate NPU architecture C++ enumerator: kDavC220
// EMPTY-SPELLING: error: NPU architecture spelling must not be empty
// EMPTY-ENUM: error: NPU architecture C++ enumerator has an empty name stem
// INVALID-ENUM: error: invalid NPU architecture C++ enumerator: kBad-Name
// RESERVED-ENUM: error: NPU architecture C++ enumerator 'kUnknown' is reserved
// RESERVED-SPELLING: error: NPU architecture spelling 'unknown' is reserved

//--- duplicate-spelling.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def AnotherArchitecture : NPUArchitecture<"dav-c220">;

//--- duplicate-enumerator.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def DavC220 : NPUArchitecture<"another-architecture">;

//--- empty-spelling.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def EmptySpelling : NPUArchitecture<"">;

//--- empty-enumerator.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def ___ : NPUArchitecture<"empty-enumerator">;

//--- invalid-enumerator.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def "Bad-Name" : NPUArchitecture<"invalid-enumerator">;

//--- reserved-enumerator.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def Unknown : NPUArchitecture<"reserved-enumerator">;

//--- reserved-spelling.td
include "bishengir/Target/NPU/NPUArchitecture.td"
def ReservedSpelling : NPUArchitecture<"unknown">;
