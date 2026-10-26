// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B1 --optimize-hivm-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B1 --optimize-hivm-pipeline='skip-hivm-bind-sub-block-pass=true' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SKIP
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend910B1 --optimize-hivm-pipeline='enable-auto-bind-sub-block=false' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLE-TILING
// RUN: bishengir-opt %s --hacc-append-device-spec=target=Ascend960 --optimize-hivm-pipeline='target=Ascend960' --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=A6

// DEFAULT: hivm-bind-sub-block{{.*}}enable-tile=true
// SKIP-NOT: hivm-bind-sub-block
// DISABLE-TILING: hivm-bind-sub-block{{.*}}enable-tile=false
// A6: hivm-bind-sub-block

module {}
