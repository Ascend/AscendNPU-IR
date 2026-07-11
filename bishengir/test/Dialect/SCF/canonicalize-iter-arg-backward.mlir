// RUN: bishengir-opt -scf-canonicalize-iter-arg -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @drop_unused_for_iter_arg
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: scf.yield %{{.*}} : index
func.func @drop_unused_for_iter_arg(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %live_next, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_all_unused_pure_for_iter_args
// CHECK-NOT: scf.for
// CHECK: return %{{.*}} : index
func.func @drop_all_unused_pure_for_iter_args(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%dead0 = %arg0, %dead1 = %arg1) -> (index, index) {
    %next0 = arith.addi %dead0, %c1 : index
    %next1 = arith.addi %dead1, %c1 : index
    scf.yield %next0, %next1 : index, index
  }
  return %c0 : index
}

// -----

// CHECK-LABEL: func.func @keep_iter_arg_used_by_store
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_iter_arg_used_by_store(%arg0: index, %arg1: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %side_next = arith.addi %side, %c1 : index
    memref.store %side_next, %out[%c0] : memref<1xindex>
    scf.yield %live_next, %side_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unused_chain_but_keep_result_chain
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK: arith.muli
// CHECK-NOT: arith.subi
// CHECK: scf.yield %{{.*}} : index
func.func @drop_unused_chain_but_keep_result_chain(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %live_add = arith.addi %live, %c1 : index
    %live_next = arith.muli %live_add, %c2 : index
    %dead_sub = arith.subi %dead, %c1 : index
    %dead_next = arith.addi %dead_sub, %c2 : index
    scf.yield %live_next, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_dead_outer_and_nested_for_iter_args
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: scf.for
// CHECK: scf.yield %{{.*}} : index
func.func @drop_dead_outer_and_nested_for_iter_args(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %inner = scf.for %j = %c0 to %c4 step %c1
        iter_args(%nested_dead = %dead) -> (index) {
      %nested_next = arith.addi %nested_dead, %c1 : index
      scf.yield %nested_next : index
    }
    scf.yield %live_next, %inner : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_outer_iter_arg_used_by_nested_for_store
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_outer_iter_arg_used_by_nested_for_store(%arg0: index, %arg1: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %inner = scf.for %j = %c0 to %c4 step %c1
        iter_args(%nested_side = %side) -> (index) {
      %nested_next = arith.addi %nested_side, %c1 : index
      memref.store %nested_next, %out[%c0] : memref<1xindex>
      scf.yield %nested_next : index
    }
    scf.yield %live_next, %inner : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unchanged_iter_arg_used_by_nested_for_bounds
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK: scf.for %{{.*}} = %{{.*}} to %arg1 step %{{.*}}
// CHECK: memref.store
func.func @drop_unchanged_iter_arg_used_by_nested_for_bounds(%arg0: index, %arg1: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %bound = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    scf.for %j = %c0 to %bound step %c1 {
      memref.store %j, %out[%c0] : memref<1xindex>
    }
    scf.yield %live_next, %bound : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_if_result_for_live_channel_drop_dead_channel
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK: scf.if
// CHECK: scf.yield %{{.*}} : index
func.func @keep_if_result_for_live_channel_drop_dead_channel(%arg0: index, %arg1: index, %cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %selected = scf.if %cond -> (index) {
      %then_next = arith.addi %live, %c1 : index
      scf.yield %then_next : index
    } else {
      %else_next = arith.addi %live, %c1 : index
      scf.yield %else_next : index
    }
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %selected, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_iter_arg_used_by_if_side_effect
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.if
// CHECK: memref.store
func.func @keep_iter_arg_used_by_if_side_effect(%arg0: index, %arg1: index, %cond: i1, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %side_next = arith.addi %side, %c1 : index
    scf.if %cond {
      memref.store %side_next, %out[%c0] : memref<1xindex>
    }
    scf.yield %live_next, %side_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unused_while_iter_arg
// CHECK: scf.while (%{{.*}} = %arg0) : (index) -> index
// CHECK: scf.condition(%{{.*}}) %{{.*}} : index
// CHECK: scf.yield %{{.*}} : index
func.func @drop_unused_while_iter_arg(%arg0: index, %arg1: index, %ub: index) -> index {
  %c1 = arith.constant 1 : index
  %res:2 = scf.while (%live = %arg0, %dead = %arg1) : (index, index) -> (index, index) {
    %cond = arith.cmpi slt, %live, %ub : index
    scf.condition(%cond) %live, %dead : index, index
  } do {
  ^bb0(%after_live: index, %after_dead: index):
    %live_next = arith.addi %after_live, %c1 : index
    %dead_next = arith.addi %after_dead, %c1 : index
    scf.yield %live_next, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_while_iter_arg_used_by_store
// CHECK: scf.while (%{{.*}} = %arg0, %{{.*}} = %arg1) : (index, index) -> (index, index)
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_while_iter_arg_used_by_store(%arg0: index, %arg1: index, %ub: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res:2 = scf.while (%live = %arg0, %side = %arg1) : (index, index) -> (index, index) {
    %cond = arith.cmpi slt, %live, %ub : index
    scf.condition(%cond) %live, %side : index, index
  } do {
  ^bb0(%after_live: index, %after_side: index):
    %live_next = arith.addi %after_live, %c1 : index
    %side_next = arith.addi %after_side, %c1 : index
    memref.store %side_next, %out[%c0] : memref<1xindex>
    scf.yield %live_next, %side_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unused_while_iter_arg_with_parent_defined_init
// CHECK: %[[START:.*]] = arith.addi %arg0, %{{.*}} : index
// CHECK: scf.while (%{{.*}} = %[[START]]) : (index) -> index
// CHECK-NOT: scf.while (%{{.*}} = %[[START]], %{{.*}} = %arg1)
// CHECK: scf.condition(%{{.*}}) %{{.*}} : index
// CHECK: scf.yield %{{.*}} : index
func.func @drop_unused_while_iter_arg_with_parent_defined_init(%arg0: index, %arg1: index, %ub: index) -> index {
  %c1 = arith.constant 1 : index
  %start = arith.addi %arg0, %c1 : index
  %res:2 = scf.while (%live = %start, %dead = %arg1) : (index, index) -> (index, index) {
    %cond = arith.cmpi slt, %live, %ub : index
    scf.condition(%cond) %live, %dead : index, index
  } do {
  ^bb0(%after_live: index, %after_dead: index):
    %live_next = arith.addi %after_live, %c1 : index
    %dead_next = arith.addi %after_dead, %c1 : index
    scf.yield %live_next, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_while_iter_arg_used_by_condition_only
// CHECK: scf.while (%{{.*}} = %arg0, %{{.*}} = %arg1) : (index, index) -> (index, index)
// CHECK: arith.cmpi slt, %{{.*}}, %arg2 : index
// CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}} : index, index
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_while_iter_arg_used_by_condition_only(%arg0: index, %arg1: index, %ub: index) -> index {
  %c1 = arith.constant 1 : index
  %res:2 = scf.while (%live = %arg0, %guard = %arg1) : (index, index) -> (index, index) {
    %cond = arith.cmpi slt, %guard, %ub : index
    scf.condition(%cond) %live, %guard : index, index
  } do {
  ^bb0(%after_live: index, %after_guard: index):
    %live_next = arith.addi %after_live, %c1 : index
    %guard_next = arith.addi %after_guard, %c1 : index
    scf.yield %live_next, %guard_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_dead_while_channel_but_keep_if_result_chain
// CHECK: scf.while (%{{.*}} = %arg0) : (index) -> index
// CHECK: scf.if
// CHECK-NOT: scf.while (%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: scf.yield %{{.*}} : index
func.func @drop_dead_while_channel_but_keep_if_result_chain(%arg0: index, %arg1: index, %ub: index, %cond: i1) -> index {
  %c1 = arith.constant 1 : index
  %res:2 = scf.while (%live = %arg0, %dead = %arg1) : (index, index) -> (index, index) {
    %loop_cond = arith.cmpi slt, %live, %ub : index
    scf.condition(%loop_cond) %live, %dead : index, index
  } do {
  ^bb0(%after_live: index, %after_dead: index):
    %selected = scf.if %cond -> (index) {
      %then_next = arith.addi %after_live, %c1 : index
      scf.yield %then_next : index
    } else {
      %else_next = arith.addi %after_live, %c1 : index
      scf.yield %else_next : index
    }
    %dead_next = arith.addi %after_dead, %c1 : index
    scf.yield %selected, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_outer_iter_arg_used_by_nested_while_store
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.while (%{{.*}} = %{{.*}}) : (index) -> index
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_outer_iter_arg_used_by_nested_while_store(%arg0: index, %arg1: index, %ub: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %inner = scf.while (%nested_side = %side) : (index) -> (index) {
      %cond = arith.cmpi slt, %nested_side, %ub : index
      scf.condition(%cond) %nested_side : index
    } do {
    ^bb0(%after_side: index):
      %nested_next = arith.addi %after_side, %c1 : index
      memref.store %nested_next, %out[%c0] : memref<1xindex>
      scf.yield %nested_next : index
    }
    scf.yield %live_next, %inner : index, index
  }
  return %res#0 : index
}

// -----

// Triggers the fixed-point keep propagation in RemoveDeadIterArgBackwardFor:
// result#0 is live, but it traces first to iter-arg#1, then to iter-arg#2 via
// tied yields in subsequent iterations. Channels #3/#4/#5 remain dead and are
// removed.
// CHECK-LABEL: func.func @drop_unused_iter_args_requires_fixed_point
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1, %{{.*}} = %arg2) -> (index, index, index) {
// CHECK: arith.addi
// CHECK-NOT: arith.addi
// CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
// CHECK-NOT: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: return %{{.*}}#0 : index
func.func @drop_unused_iter_args_requires_fixed_point(%arg0: index, %arg1: index, %arg2: index,
                                                       %arg3: index, %arg4: index, %arg5: index)
    -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:6 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%ch0 = %arg0, %ch1 = %arg1, %ch2 = %arg2,
                %dead0 = %arg3, %dead1 = %arg4, %dead2 = %arg5)
      -> (index, index, index, index, index, index) {
    %ch2_next = arith.addi %ch2, %c1 : index
    %dead0_next = arith.addi %dead0, %c1 : index
    %dead1_next = arith.addi %dead1, %c1 : index
    %dead2_next = arith.addi %dead2, %c1 : index
    scf.yield %ch1, %ch2, %ch2_next, %dead0_next, %dead1_next, %dead2_next
        : index, index, index, index, index, index
  }
  return %res#0 : index
}

// -----

// Triggers the fixed-point keep propagation in RemoveDeadIterArgBackwardWhile:
// result#0 is live, but it traces first to channel#1, then to channel#2 via
// condition/yield channel forwarding. Channels #3/#4/#5 remain dead and are
// removed.
// CHECK-LABEL: func.func @drop_unused_while_iter_args_requires_fixed_point
// CHECK: scf.while (%{{.*}} = %arg0, %{{.*}} = %arg1, %{{.*}} = %arg2) : (index, index, index) -> (index, index, index)
// CHECK-NOT: scf.while (%{{.*}} = %arg0, %{{.*}} = %arg1, %{{.*}} = %arg2, %{{.*}} = %arg3
// CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
// CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
// CHECK-NOT: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
// CHECK: return %{{.*}}#0 : index
func.func @drop_unused_while_iter_args_requires_fixed_point(%arg0: index, %arg1: index, %arg2: index,
                                                             %arg3: index, %arg4: index, %arg5: index,
                                                             %ub: index) -> index {
  %c1 = arith.constant 1 : index
  %res:6 = scf.while (%ch0 = %arg0, %ch1 = %arg1, %ch2 = %arg2,
                      %dead0 = %arg3, %dead1 = %arg4, %dead2 = %arg5)
      : (index, index, index, index, index, index)
        -> (index, index, index, index, index, index) {
    %cond = arith.cmpi slt, %ch0, %ub : index
    scf.condition(%cond) %ch1, %ch2, %ch2, %dead0, %dead1, %dead2
        : index, index, index, index, index, index
  } do {
  ^bb0(%after0: index, %after1: index, %after2: index,
       %after_dead0: index, %after_dead1: index, %after_dead2: index):
    %ch2_next = arith.addi %after2, %c1 : index
    %dead0_next = arith.addi %after_dead0, %c1 : index
    %dead1_next = arith.addi %after_dead1, %c1 : index
    %dead2_next = arith.addi %after_dead2, %c1 : index
    scf.yield %after1, %after2, %ch2_next, %dead0_next, %dead1_next, %dead2_next
        : index, index, index, index, index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_dead_for_iter_arg_through_nested_for_and_if_results
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: %{{.*}} = scf.for
// CHECK: %{{.*}} = scf.if %arg2 -> (index)
// CHECK: scf.yield %{{.*}} : index
func.func @drop_dead_for_iter_arg_through_nested_for_and_if_results(%arg0: index, %arg1: index, %cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %nested_live = scf.for %j = %c0 to %c4 step %c1
        iter_args(%inner_live = %live) -> (index) {
      %selected_live = scf.if %cond -> (index) {
        %then_live = arith.addi %inner_live, %c1 : index
        scf.yield %then_live : index
      } else {
        %else_live = arith.addi %inner_live, %c2 : index
        scf.yield %else_live : index
      }
      scf.yield %selected_live : index
    }
    %nested_dead = scf.for %k = %c0 to %c4 step %c1
        iter_args(%inner_dead = %dead) -> (index) {
      %selected_dead = scf.if %cond -> (index) {
        %then_dead = arith.addi %inner_dead, %c1 : index
        scf.yield %then_dead : index
      } else {
        %else_dead = arith.addi %inner_dead, %c2 : index
        scf.yield %else_dead : index
      }
      scf.yield %selected_dead : index
    }
    %live_next = scf.if %cond -> (index) {
      scf.yield %nested_live : index
    } else {
      %fallback_live = arith.addi %live, %c1 : index
      scf.yield %fallback_live : index
    }
    scf.yield %live_next, %nested_dead : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_outer_iter_arg_used_by_nested_for_if_result_store
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK: scf.if %arg2 -> (index)
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_outer_iter_arg_used_by_nested_for_if_result_store(%arg0: index, %arg1: index, %cond: i1, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %nested_side = scf.for %j = %c0 to %c4 step %c1
        iter_args(%inner_side = %side) -> (index) {
      %selected_side = scf.if %cond -> (index) {
        %then_side = arith.addi %inner_side, %c1 : index
        scf.yield %then_side : index
      } else {
        %else_side = arith.addi %inner_side, %c2 : index
        scf.yield %else_side : index
      }
      memref.store %selected_side, %out[%c0] : memref<1xindex>
      scf.yield %selected_side : index
    }
    scf.yield %live_next, %nested_side : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @keep_outer_iter_arg_used_by_nested_if_branch_stores
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK: scf.if %arg2 -> (index)
// CHECK: memref.store
// CHECK: memref.store
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_outer_iter_arg_used_by_nested_if_branch_stores(%arg0: index, %arg1: index, %cond: i1, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %nested_side = scf.for %j = %c0 to %c4 step %c1
        iter_args(%inner_side = %side) -> (index) {
      %selected_side = scf.if %cond -> (index) {
        %then_side = arith.addi %inner_side, %c1 : index
        memref.store %then_side, %out[%c0] : memref<1xindex>
        scf.yield %then_side : index
      } else {
        %else_side = arith.addi %inner_side, %c2 : index
        memref.store %else_side, %out[%c0] : memref<1xindex>
        scf.yield %else_side : index
      }
      scf.yield %selected_side : index
    }
    scf.yield %live_next, %nested_side : index, index
  }
  return %res#0 : index
}

// -----

func.func private @side_effect_sink(index)

// CHECK-LABEL: func.func @keep_outer_iter_arg_used_by_nested_if_branch_calls
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1) -> (index, index) {
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK: scf.if %arg2 -> (index)
// CHECK: func.call @side_effect_sink
// CHECK: func.call @side_effect_sink
// CHECK: scf.yield %{{.*}}, %{{.*}} : index, index
func.func @keep_outer_iter_arg_used_by_nested_if_branch_calls(%arg0: index, %arg1: index, %cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %side = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    %nested_side = scf.for %j = %c0 to %c4 step %c1
        iter_args(%inner_side = %side) -> (index) {
      %selected_side = scf.if %cond -> (index) {
        %then_side = arith.addi %inner_side, %c1 : index
        func.call @side_effect_sink(%then_side) : (index) -> ()
        scf.yield %then_side : index
      } else {
        %else_side = arith.addi %inner_side, %c2 : index
        func.call @side_effect_sink(%else_side) : (index) -> ()
        scf.yield %else_side : index
      }
      scf.yield %selected_side : index
    }
    scf.yield %live_next, %nested_side : index, index
  }
  return %res#0 : index
}


// -----

// Regression: the backward dead-iter-arg pattern must not hard-assert when a
// nested loop result that would be dropped is still consumed by a live
// (non-terminator) op. This mirrors a mix-mode / cv-pipeline kernel: the outer
// loop's own results are dead (so the pattern tries to prune both channels),
// but the nested loop's result feeds an annotation.mark in the outer body
// (besides being yielded). The nested result does not trace back to the outer
// iter-arg -- it is produced from a freshly written memref -- so the backward
// trace leaves the channel removable while a live op still uses it. The pass
// must bail and leave the loop nest intact instead of aborting.
// CHECK-LABEL: func.func @bail_nested_result_used_by_live_op
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1xf32>) {
// CHECK:   scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1xf32>) {
// CHECK:     memref.store
// CHECK:     scf.yield %{{.*}} : tensor<1xf32>
// CHECK:   }
// CHECK:   annotation.mark
// CHECK:   scf.yield %{{.*}} : tensor<1xf32>
// CHECK: }
func.func @bail_nested_result_used_by_live_op(%lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %e = tensor.empty() : tensor<1xf32>
  %outer = scf.for %i = %lb to %ub step %step iter_args(%a = %e) -> (tensor<1xf32>) {
    %n = scf.for %j = %lb to %ub step %step iter_args(%x = %a) -> (tensor<1xf32>) {
      %m = memref.alloc() : memref<1xf32>
      memref.store %cst, %m[%c0] : memref<1xf32>
      %t = bufferization.to_tensor %m restrict writable : memref<1xf32>
      scf.yield %t : tensor<1xf32>
    }
    annotation.mark %n {cv_pipeline_lazy_load = true} : tensor<1xf32>
    scf.yield %n : tensor<1xf32>
  }
  return
}

// -----

// CHECK-LABEL: func.func @drop_dead_for_iter_arg_through_scope_result
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK: %[[SCOPE:.*]] = scope.scope
// CHECK: scope.return %{{.*}} : index
// CHECK-NOT: scope.return %{{.*}}, %{{.*}} : index, index
// CHECK: scf.yield %[[SCOPE]] : index
func.func @drop_dead_for_iter_arg_through_scope_result(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %scope:2 = scope.scope : () -> (index, index) {
      %live_next = arith.addi %live, %c1 : index
      %dead_next = arith.addi %dead, %c1 : index
      scope.return %live_next, %dead_next : index, index
    }
    scf.yield %scope#0, %scope#1 : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_dead_while_iter_arg_through_scope_result
// CHECK: scf.while (%{{.*}} = %arg0) : (index) -> index
// CHECK: %[[SCOPE:.*]] = scope.scope
// CHECK: scope.return %{{.*}} : index
// CHECK-NOT: scope.return %{{.*}}, %{{.*}} : index, index
// CHECK: scf.yield %[[SCOPE]] : index
func.func @drop_dead_while_iter_arg_through_scope_result(%arg0: index, %arg1: index, %ub: index) -> index {
  %c1 = arith.constant 1 : index
  %res:2 = scf.while (%live = %arg0, %dead = %arg1) : (index, index) -> (index, index) {
    %cond = arith.cmpi slt, %live, %ub : index
    scf.condition(%cond) %live, %dead : index, index
  } do {
  ^bb0(%after_live: index, %after_dead: index):
    %scope:2 = scope.scope : () -> (index, index) {
      %live_next = arith.addi %after_live, %c1 : index
      %dead_next = arith.addi %after_dead, %c1 : index
      scope.return %live_next, %dead_next : index, index
    }
    scf.yield %scope#0, %scope#1 : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @scope_no_results_survives_dead_iter_arg_drop
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: scope.scope : () -> () {
// CHECK: memref.store
// CHECK: scope.return
func.func @scope_no_results_survives_dead_iter_arg_drop(%arg0: index, %arg1: index, %out: memref<1xindex>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %live_next = arith.addi %live, %c1 : index
    scope.scope : () -> () {
      memref.store %live_next, %out[%c0] : memref<1xindex>
      scope.return
    }
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %live_next, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @scope_results_kept_drop_inner_dead_use
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: %[[SCOPE:.*]] = scope.scope : () -> index {
// CHECK: arith.addi
// CHECK-NOT: arith.subi
// CHECK: scope.return %{{.*}} : index
// CHECK: scf.yield %[[SCOPE]] : index
func.func @scope_results_kept_drop_inner_dead_use(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %scope_live = scope.scope : () -> index {
      %live_step0 = arith.addi %live, %c1 : index
      %dead_unused = arith.subi %dead, %c1 : index
      %live_step1 = arith.addi %live_step0, %c1 : index
      scope.return %live_step1 : index
    }
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %scope_live, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @loop_within_scope_within_loop
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: %[[SCOPE:.*]] = scope.scope : () -> index {
// CHECK: %[[INNER:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK: scope.return %[[INNER]] : index
// CHECK: scf.yield %[[SCOPE]] : index
func.func @loop_within_scope_within_loop(%arg0: index, %arg1: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %scope_live = scope.scope : () -> index {
      %inner = scf.for %j = %c0 to %c4 step %c1
          iter_args(%inner_live = %live) -> (index) {
        %inner_next = arith.addi %inner_live, %c1 : index
        scf.yield %inner_next : index
      }
      scope.return %inner : index
    }
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %scope_live, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @if_within_scope_within_loop
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %arg0) -> (index) {
// CHECK-NOT: iter_args(%{{.*}} = %arg0, %{{.*}} = %arg1)
// CHECK: scope.scope : () -> index {
// CHECK: scf.if %arg2 -> (index)
// CHECK: scope.return %{{.*}} : index
func.func @if_within_scope_within_loop(%arg0: index, %arg1: index, %cond: i1) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %res:2 = scf.for %iv = %c0 to %c4 step %c1
      iter_args(%live = %arg0, %dead = %arg1) -> (index, index) {
    %scope_live = scope.scope : () -> index {
      %selected = scf.if %cond -> (index) {
        %then_v = arith.addi %live, %c1 : index
        scf.yield %then_v : index
      } else {
        %else_v = arith.addi %live, %c2 : index
        scf.yield %else_v : index
      }
      scope.return %selected : index
    }
    %dead_next = arith.addi %dead, %c1 : index
    scf.yield %scope_live, %dead_next : index, index
  }
  return %res#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unused_scope_result_outer_values
// CHECK-NOT: scope.scope
// CHECK: return %arg0 : index
func.func @drop_unused_scope_result_outer_values(%arg0: index, %arg1: index) -> index {
  %scope:2 = scope.scope : () -> (index, index) {
    scope.return %arg0, %arg1 : index, index
  }
  return %scope#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_unused_scope_result_channel
// CHECK: %[[SCOPE:.*]] = scope.scope : () -> index {
// CHECK: arith.addi
// CHECK-NOT: arith.subi
// CHECK: scope.return %{{.*}} : index
// CHECK-NOT: scope.return %{{.*}}, %{{.*}} : index, index
// CHECK: return %[[SCOPE]] : index
func.func @drop_unused_scope_result_channel(%arg0: index, %arg1: index) -> index {
  %scope:2 = scope.scope : () -> (index, index) {
    %live = arith.addi %arg0, %arg1 : index
    %dead = arith.subi %arg0, %arg1 : index
    scope.return %live, %dead : index, index
  }
  return %scope#0 : index
}

// -----

// CHECK-LABEL: func.func @drop_all_unused_scope_results_keep_side_effects
// CHECK-NOT: scope.scope : () -> (index
// CHECK: scope.scope : () -> () {
// CHECK: arith.addi
// CHECK-NOT: arith.subi
// CHECK: memref.store
// CHECK: scope.return
func.func @drop_all_unused_scope_results_keep_side_effects(%arg0: index, %arg1: index,
                                                            %out: memref<1xindex>) {
  %c0 = arith.constant 0 : index
  %scope:2 = scope.scope : () -> (index, index) {
    %stored = arith.addi %arg0, %arg1 : index
    memref.store %stored, %out[%c0] : memref<1xindex>
    %dead = arith.subi %arg0, %arg1 : index
    scope.return %stored, %dead : index, index
  }
  return
}

// -----

// CHECK-LABEL: func.func @canonicalize_scope_result_defined_outside
// CHECK: %[[SCOPE:.*]] = scope.scope : () -> index {
// CHECK-NOT: scope.scope : () -> (index, index)
// CHECK: scope.return %{{.*}} : index
// CHECK-NOT: scope.return %arg0, %{{.*}} : index, index
// CHECK: %[[SUM:.*]] = arith.addi %arg0, %c1 : index
// CHECK: %[[MIX:.*]] = arith.addi %[[SUM]], %[[SCOPE]] : index
func.func @canonicalize_scope_result_defined_outside(%arg0: index, %arg1: index) -> index {
  %c1 = arith.constant 1 : index
  %scope:2 = scope.scope : () -> (index, index) {
    %inner = arith.addi %arg0, %arg1 : index
    scope.return %arg0, %inner : index, index
  }
  %sum = arith.addi %scope#0, %c1 : index
  %mix = arith.addi %sum, %scope#1 : index
  return %mix : index
}
