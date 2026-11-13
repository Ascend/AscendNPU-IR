// RUN: bishengir-opt %s -split-input-file -verify-diagnostics

func.func @pack_part(%p: vector<128xi1>) {
  // expected-error @+1 {{part must be 0}}
  %r = ave.hir.preg.pack %p, 2 : vector<128xi1> -> vector<256xi1>
  return
}

// -----

func.func @pack_size(%p: vector<64xi1>) {
  // expected-error @+1 {{factor-of-two logical lane change}}
  %r = ave.hir.preg.pack %p, 0 : vector<64xi1> -> vector<256xi1>
  return
}

// -----

func.func @unpack_size(%p: vector<128xi1>) {
  // expected-error @+1 {{factor-of-two logical lane change}}
  %r = ave.hir.preg.unpack %p, 0 : vector<128xi1> -> vector<128xi1>
  return
}

// -----

func.func @short_pred(%p: vector<32xi1>) {
  // expected-error @+1 {{fixed 64/128/256-lane predicates}}
  %r = ave.hir.preg.pack %p, 0 : vector<32xi1> -> vector<64xi1>
  return
}

// -----

func.func @scalable_pred(%p: vector<[64]xi1>) {
  // expected-error @+1 {{fixed 64/128/256-lane predicates}}
  %r = ave.hir.preg.pack %p, 0 : vector<[64]xi1> -> vector<[128]xi1>
  return
}

// -----

func.func @short_interleave(%p: vector<32xi1>) {
  // expected-error @+1 {{fixed 64/128/256-lane predicates}}
  %a, %b = ave.hir.preg.interleave %p, %p : vector<32xi1>
  return
}

// -----

func.func @short_deinterleave(%p: vector<32xi1>) {
  // expected-error @+1 {{fixed 64/128/256-lane predicates}}
  %a, %b = ave.hir.preg.deinterleave %p, %p : vector<32xi1>
  return
}

// -----

func.func @gather_short(%v: vector<32xi32>, %idx: vector<32xi32>, %p: vector<32xi1>) {
  // expected-error @+1 {{full 8/16/32-bit register}}
  %r = ave.hir.vselr %v, %idx, %p : vector<32xi32>, vector<32xi32>, vector<32xi1>
  return
}

// -----

func.func @gather_index(%v: vector<128xf16>, %idx: vector<64xi32>, %p: vector<128xi1>) {
  // expected-error @+1 {{indices must be signless integers matching source width and lanes}}
  %r = ave.hir.vselr %v, %idx, %p : vector<128xf16>, vector<64xi32>, vector<128xi1>
  return
}

// -----

func.func @gather_mask(%v: vector<64xi32>, %idx: vector<64xi32>, %p: vector<128xi1>) {
  // expected-error @+1 {{mask must have the source logical lane count}}
  %r = ave.hir.vselr %v, %idx, %p : vector<64xi32>, vector<64xi32>, vector<128xi1>
  return
}

// -----

func.func @group_widen(%v: vector<128xi16>, %p: vector<128xi1>) {
  // expected-error @+1 {{i16 add returns vector<64xi32>}}
  %r = ave.hir.group_reduction <add> %v, %p : vector<128xi16>, vector<128xi1> -> vector<128xi16>
  return
}

// -----

func.func @group_kind(%v: vector<64xi32>, %p: vector<64xi1>) {
  // expected-error @+1 {{supports only add/min/max}}
  %r = ave.hir.group_reduction <umax> %v, %p : vector<64xi32>, vector<64xi1> -> vector<64xi32>
  return
}

// -----

func.func @group_mask(%v: vector<128xf16>, %p: vector<64xi1>) {
  // expected-error @+1 {{mask must have the source logical lane count}}
  %r = ave.hir.group_reduction <max> %v, %p : vector<128xf16>, vector<64xi1> -> vector<128xf16>
  return
}

// -----

func.func @group_i8(%v: vector<256xi8>, %p: vector<256xi1>) {
  // expected-error @+1 {{full i16/i32/f16/f32 register}}
  %r = ave.hir.group_reduction <add> %v, %p : vector<256xi8>, vector<256xi1> -> vector<256xi8>
  return
}

// -----

func.func @gather_oob_even_if_inactive(%src: vector<64xf32>) {
  %idx = arith.constant dense<64> : vector<64xi32>
  %mask = ave.hir.pge <ALLF> : vector<64xi1>
  // expected-error @+1 {{every index must be in [0, source lanes), including inactive lanes}}
  %r = ave.hir.vselr %src, %idx, %mask : vector<64xf32>, vector<64xi32>, vector<64xi1>
  return
}

// -----

func.func @group_unsigned_float(%v: vector<64xf32>, %p: vector<64xi1>) {
  // expected-error @+1 {{floating-point sources require cast_signed}}
  %r = ave.hir.group_reduction <max> %v, %p {cast = #hivm.cast<cast_unsigned>} : vector<64xf32>, vector<64xi1> -> vector<64xf32>
  return
}

// -----

func.func @group_result_type(%v: vector<128xi16>, %p: vector<128xi1>) {
  // expected-error @+1 {{result must preserve source type}}
  %r = ave.hir.group_reduction <min> %v, %p : vector<128xi16>, vector<128xi1> -> vector<64xi32>
  return
}

// -----

func.func @gather_index_element(%v: vector<64xindex>, %i: vector<64xi32>, %p: vector<64xi1>) {
  // expected-error @+1 {{requires a full 8/16/32-bit register}}
  %r = ave.hir.vselr %v, %i, %p : vector<64xindex>, vector<64xi32>, vector<64xi1>
  return
}

// -----

func.func @group_index_element(%v: vector<64xindex>, %p: vector<64xi1>) {
  // expected-error @+1 {{requires a full i16/i32/f16/f32 register}}
  %r = ave.hir.group_reduction <add> %v, %p : vector<64xindex>, vector<64xi1> -> vector<64xindex>
  return
}
