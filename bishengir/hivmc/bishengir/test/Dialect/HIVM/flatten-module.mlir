// RUN: bishengir-opt %s -hivm-flatten-module -split-input-file -verify-diagnostics | FileCheck %s


// CHECK: @main
// CHECK-NOT: submod
module @main {
   llvm.func @fA(){
     llvm.call @foo() : () -> ()
    llvm.return
   }
   llvm.func @foo()

  module @submod attributes {hacc.simt_module}{
    llvm.func @foo(){
        llvm.return
    }
  }
}

// -----

// CHECK: @main
// CHECK-NOT: submod
module @nested_parallel {
  module @main {
    llvm.func @fA(){
        llvm.call @foo() : () -> ()
        llvm.return
    }
    llvm.func @foo()
  }

  module @submod attributes {hacc.simt_module} {
    llvm.func @foo(){
        llvm.return
    }
  }
}

// -----

// CHECK: @main
// CHECK-NOT: submod
module @func_rename_norm {
  module @main {
    llvm.func @fA(){
        llvm.return
    }
  }

  module @submod attributes {hacc.simt_module} {
    llvm.func @fA(){
        llvm.return
    }
  }
}

// -----

// CHECK: @main
// CHECK-NOT: submod
module @func_rename_fix_caller {
  module @main {
    llvm.func @fA(){
        llvm.return
    }
  }

  module @submod attributes {hacc.simt_module}{
    llvm.func @main(){
        llvm.call @fA():()->()
        llvm.return
    }
    llvm.func @fA(){
        llvm.return
    }
  }
}

// -----

// CHECK: @main
// CHECK: @_mlir_ciface_a
module @func_merge_ciface {
  module @main {
    llvm.func @_mlir_ciface_a()
  }

  module @submod attributes {hacc.simt_module}{
    llvm.func @_mlir_ciface_a()
  }
}

// -----

// CHECK: @main
module @kernel {
  module @main attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
    llvm.func @entry(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>){
        // CHECK-NOT: llvm.call @simt_func
        // CHECK: hivm_regbaseintrins.intrins.launch_func
        llvm.call @simt_func(%arg0, %arg1, %arg2) {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
        llvm.return
    }
    llvm.func @simt_func(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>}{
        llvm.return
    }
  }

  module @simt_mod attributes {hacc.simt_module} {
    llvm.func @simt_func(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">}{
      %0 = llvm.mlir.constant(128 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %2 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
      hivm_regbaseintrins.intrins.launch_func @simt_func_vf_simt threads in (%0, %1, %1) args(%arg0, %arg1, %arg2) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>
      llvm.return
    }
    llvm.func @simt_func_vf_simt(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<128>, nvvm.kernel = 1 : ui1}{
      llvm.return
    }
  }
}

// -----

// CHECK: @main
// CHECK: llvm.urem
module @kernel_ext {
  module @main attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
    llvm.func @entry(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>){
        // CHECK-NOT: llvm.call @simt_func
        // CHECK: hivm_regbaseintrins.intrins.launch_func
        %0 = llvm.mlir.constant(612 : i64) : i64
        llvm.call @simt_func(%arg0, %arg1, %arg2) {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} : (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
        llvm.return
    }
    llvm.func @simt_func(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hivm.vf_mode = #hivm.vf_mode<SIMT>}{
        llvm.return
    }
  }

  module @simt_mod attributes {hacc.simt_module} {
    llvm.func @simt_func(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>, hivm_regbaseintrins.kernel, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">}{
      %0 = llvm.mlir.constant(128 : i64) : i64
      %1 = llvm.mlir.constant(1 : i64) : i64
      %2 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
      %3 = llvm.mlir.constant(32 : i64) : i64
      %4 = llvm.mlir.constant(256 : i64) : i64
      %5 = llvm.mlir.constant(64: i64) : i64
      %6 = llvm.urem %4, %5 : i64

      hivm_regbaseintrins.intrins.launch_func @simt_func_vf_simt threads in (%0, %1, %1) args(%arg0, %arg1, %arg2, %3, %6) : !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64
      llvm.return
    }
    llvm.func @simt_func_vf_simt(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: i64, %arg4: i64) attributes {hivm_regbaseintrins.cconv = #hivm_regbaseintrins.simt_entry<128>, nvvm.kernel = 1 : ui1}{
      llvm.return
    }
  }
}
