// RUN: not bishengir-opt %s -mlir-print-stacktrace-on-diagnostic 2>&1 | FileCheck %s

// CHECK: error: operation being parsed with an unregistered dialect
// CHECK: note: diagnostic emitted with trace:
// CHECK: #{{[0-9]+}} {{.*}}PrintStackTrace

"unknown.op"() : () -> ()
