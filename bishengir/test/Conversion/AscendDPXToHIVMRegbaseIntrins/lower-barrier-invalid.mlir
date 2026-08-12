// RUN: bishengir-opt %s -convert-ascend-dpx-to-hivmregbaseintrins -verify-diagnostics

// expected-error@below {{super-block barrier requires numWarps * superBlockFactor <= 64, but got numWarps=4 and superBlockFactor=32 (totalWarps=128)}}
module attributes {
  "ttg.num-warps" = 4 : i32,
  "ttg.super-block-factor" = 32 : ui32,
  "ttg.super-block-barrier" = true
} {
}
