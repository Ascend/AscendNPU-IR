// RUN: not bishengir-opt %s -hacc-append-device-spec=target=Ascend960 -convert-hivmave-to-ave-intrin 2>&1 | FileCheck %s
// RUN: not bishengir-opt %s -convert-hivmave-to-ave-intrin 2>&1 | FileCheck %s
// CHECK: group reduction currently supports only Ascend950
func.func @unsupported(%x: vector<64xf32>, %mask: vector<64xi1>) -> vector<64xf32> {
  %r = ave.hir.group_reduction <add> %x, %mask : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return %r : vector<64xf32>
}
