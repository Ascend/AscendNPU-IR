// RUN: bishengir-compile --help | FileCheck %s
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, 0), (l1, 2), (l0c, 1), (ub, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, 2147483648), (l1, 2), (l0c, 1), (ub, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=TOO-LARGE
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, 2), (gm, 3), (l0c, 1), (ub, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(global, 2), (l1, 2), (l0c, 1), (ub, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, -1), (l1, 2), (l0c, 1), (ub, 2)]' \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   2>&1 | FileCheck %s --check-prefix=NEGATIVE
// RUN: not bishengir-compile %s \
// RUN:   --limit-auto-multi-buffer-only-for-local-buffer=true \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   >%t 2>&1; FileCheck %s --check-prefix=DEPRECATED < %t
// RUN: not bishengir-compile %s \
// RUN:   --multibuffer-mode='[(gm, 1), (l1, 2), (l0c, 1), (ub, 1)]' \
// RUN:   --limit-auto-multi-buffer-buffer=no-limit \
// RUN:   --enable-hivm-compile=false --enable-hfusion-compile=false \
// RUN:   >%t 2>&1; FileCheck %s --check-prefix=NEW-WINS < %t

// CHECK-NOT: BiShengIR
// CHECK: OVERVIEW: BiShengIR Compile Tool
// CHECK: OPTIONS:
// CHECK: BiShengIR DFX Control Options:
// CHECK: BiShengIR Feature Control Options:
// CHECK: BiShengIR General Optimization Options:
// CHECK: BiShengIR HFusion Optimization Options:
// CHECK: BiShengIR HIVM Optimization Options:
// CHECK: --enable-ave-loop-optimize
// CHECK: --multibuffer-mode
// CHECK: BiShengIR SIMT Optimization Options:
// CHECK: --enable-simt-device-debug
// CHECK: BiShengIR Target Options:
// CHECK: Options Shared with HIVMC:
// CHECK-NOT: BiShengIR

// ZERO: invalid --multibuffer-mode
// ZERO: must be >= 1
// INCOMPLETE: invalid --multibuffer-mode
// INCOMPLETE: must specify all of gm, l1, l0c, and ub
// TOO-LARGE: invalid --multibuffer-mode
// TOO-LARGE: must fit in a signed 32-bit annotation
// DUPLICATE: invalid --multibuffer-mode
// DUPLICATE: duplicate level 'gm'
// UNKNOWN: invalid --multibuffer-mode
// UNKNOWN: unknown memory level 'global'
// NEGATIVE: invalid --multibuffer-mode
// NEGATIVE: invalid multi-buffer count '-1'
// DEPRECATED-DAG: deprecated
// DEPRECATED-DAG: BiShengIR 1.4.0
// DEPRECATED-DAG: --multibuffer-mode=
// NEW-WINS-DAG: deprecated and ignored when --multibuffer-mode is set
// NEW-WINS-DAG: --multibuffer-mode="[(gm, 1), (l1, 2), (l0c, 1), (ub, 1)]"

module {
  func.func @main() {
    return
  }
}
