// RUN: bishengir-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test Address Space Attribute
//===----------------------------------------------------------------------===//

func.func @address_space() {
  "test.address_space"() {
    // CHECK: #hivm.address_space<ca>
    ca = #hivm.address_space<ca>,
    // CHECK: #hivm.address_space<cb>
    cb = #hivm.address_space<cb>,
    // CHECK: #hivm.address_space<cbuf>
    cbuf = #hivm.address_space<cbuf>,
    // CHECK: #hivm.address_space<cc>
    cc = #hivm.address_space<cc>,
    // CHECK: #hivm.address_space<gm>
    gm = #hivm.address_space<gm>,
    // CHECK: #hivm.address_space<ub>
    ub = #hivm.address_space<ub>,
    // CHECK: #hivm.address_space<zero>
    zero = #hivm.address_space<zero>
  } : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// Test Data Layout Attribute
//===----------------------------------------------------------------------===//

func.func @data_layout() {
  "test.data_layout"() {
    // CHECK: #hivm.data_layout<ND>
    ND = #hivm.data_layout<ND>,
    // CHECK: #hivm.data_layout<dotA_ND, transpose = true>
    dotA_ND_transpose = #hivm.data_layout<dotA_ND, transpose = true>,
    // CHECK: #hivm.data_layout<dotA_ND, transpose = false>
    dotA_ND_transpose_false = #hivm.data_layout<dotA_ND, transpose = false>,
    // CHECK: #hivm.data_layout<dotB_ND, transpose = true>
    dotB_ND_transpose = #hivm.data_layout<dotB_ND, transpose = true>,
    // CHECK: #hivm.data_layout<dotB_ND, transpose = false>
    dotB_ND_transpose_false = #hivm.data_layout<dotB_ND, transpose = false>,
    // CHECK: #hivm.data_layout<dotC_ND>
    dotC_ND = #hivm.data_layout<dotC_ND>,
    // CHECK: #hivm.data_layout<nZ>
    nZ = #hivm.data_layout<nZ>,
    // CHECK: #hivm.data_layout<zN>
    zN = #hivm.data_layout<zN>
  } : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// Test PIPE Attribute
//===----------------------------------------------------------------------===//

func.func @pipe() {
  "test.pipe"() {
  // CHECK: #hivm.pipe<PIPE_ALL>
  pipe_all        = #hivm.pipe<PIPE_ALL>,
  // CHECK: #hivm.pipe<PIPE_FIX>
  pipe_fix        = #hivm.pipe<PIPE_FIX>,
  // CHECK: #hivm.pipe<PIPE_M>
  pipe_m          = #hivm.pipe<PIPE_M>,
  // CHECK: #hivm.pipe<PIPE_MTE1>
  pipe_mte1       = #hivm.pipe<PIPE_MTE1>,
  // CHECK: #hivm.pipe<PIPE_MTE2>
  pipe_mte2       = #hivm.pipe<PIPE_MTE2>,
  // CHECK: #hivm.pipe<PIPE_MTE3>
  pipe_mte3       = #hivm.pipe<PIPE_MTE3>,
  // CHECK: #hivm.pipe<PIPE_MTE4>
  pipe_mte4       = #hivm.pipe<PIPE_MTE4>,
  // CHECK: #hivm.pipe<PIPE_MTE5>
  pipe_mte5       = #hivm.pipe<PIPE_MTE5>,
  // CHECK: #hivm.pipe<PIPE_S>
  pipe_s          = #hivm.pipe<PIPE_S>,
  // CHECK: #hivm.pipe<PIPE_UNASSIGNED>
  pipe_unassigned = #hivm.pipe<PIPE_UNASSIGNED>,
  // CHECK: #hivm.pipe<PIPE_V>
  pipe_v          = #hivm.pipe<PIPE_V>,
  // CHECK: #hivm.pipe<PIPE_V2>
  pipe_v2         = #hivm.pipe<PIPE_V2>
  } : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// Test EVENT_ID Attribute
//===----------------------------------------------------------------------===//

func.func @event() {
  "test.event"() {
  // CHECK: #hivm.event<EVENT_ID0>
  eventID0          = #hivm.event<EVENT_ID0>,
  // CHECK: #hivm.event<EVENT_ID1>
  eventID1          = #hivm.event<EVENT_ID1>,
  // CHECK: #hivm.event<EVENT_ID2>
  eventID2          = #hivm.event<EVENT_ID2>,
  // CHECK: #hivm.event<EVENT_ID3>
  eventID3          = #hivm.event<EVENT_ID3>,
  // CHECK: #hivm.event<EVENT_ID4>
  eventID4          = #hivm.event<EVENT_ID4>,
  // CHECK: #hivm.event<EVENT_ID5>
  eventID5          = #hivm.event<EVENT_ID5>,
  // CHECK: #hivm.event<EVENT_ID6>
  eventID6          = #hivm.event<EVENT_ID6>,
  // CHECK: #hivm.event<EVENT_ID7>
  eventID7          = #hivm.event<EVENT_ID7>
  } : () -> ()

  return
}

//===----------------------------------------------------------------------===//
// Test Block Mapping Attribute
//===----------------------------------------------------------------------===//

func.func @block_mapping() {
  "test.block_mapping"() {
    // CHECK: #hivm.block<linear_dim = 0>
    x = #hivm.block<linear_dim = 0>,
    // CHECK: #hivm.block
    y = #hivm.block
  } : () -> ()

  return
}
