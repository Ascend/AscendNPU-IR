// RUN: bishengir-opt -pass-pipeline="builtin.module(canonicalize-module)" %s -split-input-file -verify-diagnostics | FileCheck %s

module {
  // CHECK: module {
  module {
    func.func @func_0(%arg0: memref<8xf32>) {
      return
    }
    func.func @func_1(%arg0: memref<8xf32>){
      return
    }
  }
  // CHECK: }
  module {
  }
}

// -----
// CHECK: module {
module {
  // CHECK: module {
  module {
    func.func @func_0(%arg0: memref<8xf32>) {
      return
    }
  }
  // CHECK: module {
  module {
    func.func @func_1(%arg0: memref<8xf32>) {
      return
    }
    // CHECK: }
    module {
    }
  }
  //CHECK: }
  module {
    module {
    }
  }
}

// -----
// CHECK: module attributes {test.attr1, test.attr2} {
// CHECK:   func.func @func_0(%arg0: memref<8xf32>) {
// CHECK:     return
// CHECK:   }
// CHECK: }
module {
  module attributes {test.attr1} {
    module attributes {test.attr2} {
      func.func @func_0(%arg0: memref<8xf32>) {
        return
      }
    }
  }
}

// -----
//CHECK: module attributes {test.attr1} {
//CHECK:   module attributes {test.attr2} {
//CHECK:     func.func @func_0(%arg0: memref<8xf32>) {
//CHECK:       return
//CHECK:     }
//CHECK:   }
//CHECK:   module attributes {test.attr3} {
//CHECK:     func.func @func_0(%arg0: memref<8xf32>) {
//CHECK:      return
//CHECK:     }
//CHECK:   }
//CHECK: }

module {
  module attributes {test.attr1} {
    module attributes {test.attr2} {
      func.func @func_0(%arg0: memref<8xf32>) {
        return
      }
    }
    module {
      module attributes {test.attr3} {
        func.func @func_0(%arg0: memref<8xf32>) {
          return
        }
      }
    }
    
  }
}
