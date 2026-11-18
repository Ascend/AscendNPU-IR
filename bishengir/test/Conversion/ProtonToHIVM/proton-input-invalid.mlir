// RUN: bishengir-opt %s -convert-proton-to-hivm -verify-diagnostics

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  // expected-error @+1 {{A5 Proton records must be nested in a function}}
  proton.record start "outer"
}
