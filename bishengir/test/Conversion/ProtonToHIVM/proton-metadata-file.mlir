// RUN: split-file %s %t
// RUN: bishengir-opt %t/nested.mlir -convert-proton-to-hivm -get-proton-metadata="proton-metadata-output=%t/nested.json" -o /dev/null
// RUN: FileCheck %s < %t/nested.json
// RUN: bishengir-opt %t/no-records.mlir -convert-proton-to-hivm -get-proton-metadata="proton-metadata-output=%t/absent.json" -o /dev/null
// RUN: not test -f %t/absent.json

// outer is scope 0. inner is nested in it, so its parent is 0.
// Region IDs are exported explicitly: start = id|0x400, end = id|0xC00.
// CHECK: [{"proton":{
// CHECK-SAME: "format_version":1,
// CHECK-SAME: "scopes":[
// CHECK-SAME: {"end_region_id":3072,"name":"outer","parent_id":-1,"start_region_id":1024},
// CHECK-SAME: {"end_region_id":3073,"name":"inner","parent_id":0,"start_region_id":1025}]}}]

//--- nested.mlir

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @nested() {
    proton.record start "outer"
    proton.record start "inner"
    proton.record end "inner"
    proton.record end "outer"
    return
  }
}

//--- no-records.mlir

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @no_records() {
    return
  }
}
