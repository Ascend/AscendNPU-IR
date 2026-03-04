// RUN: bishengir-opt %s -optimize-loads | FileCheck %s

// CHECK-LABEL: tt.func public @triton_unk_fused_cat_0

// the first load
// condition is mask & cond3 & cond2 & cond1
// CHECK:    %[[TMP0:.*]] = arith.andi %[[COND3:.*]], %[[COND2:.*]] : tensor<8x512xi1>
// CHECK:    %[[TMP1:.*]] = arith.andi %[[TMP0]], %[[COND1:.*]] : tensor<8x512xi1>
// CHECK:    %[[MASK1:.*]] = arith.andi %[[TMP1:.*]], %[[MASK:.*]] : tensor<8x512xi1>

// CHECK:    %[[VAL1:.*]] = tt.load %{{.*}}, %[[MASK1]], %{{.*}} : tensor<8x512x!tt.ptr<f32>>

// the second load
// condition is mask & cond3 & cond2 & ~cond1
//            = mask & cond3 & cond2 & (cond1 XOR true)
// CHECK:    %[[TMP0:.*]] = arith.andi %[[COND3]], %[[COND2]] : tensor<8x512xi1>
// CHECK:    %[[TMP1:.*]] = arith.xori %[[COND1]], %[[TRUE:.*]] : tensor<8x512xi1>
// CHECK:    %[[TMP2:.*]] = arith.andi %[[TMP0]], %[[TMP1]] : tensor<8x512xi1>
// CHECK:    %[[MASK2:.*]] = arith.andi %[[TMP2]], %[[MASK]] : tensor<8x512xi1>
// CHECK:    %[[VAL2:.*]] = tt.load %{{.*}}, %[[MASK2]], %{{.*}} : tensor<8x512x!tt.ptr<f32>>

// CHECK:    %[[VAL5:.*]] = arith.select %[[COND1]], %[[VAL1]], %[[VAL2]] : tensor<8x512xi1>, tensor<8x512xf32>

// the third load
// condition is mask & cond3 & ~cond2
//            = mask & cond3 & (cond2 XOR true)
// CHECK:    %[[TMP0:.*]] = arith.xori %[[COND2]], %[[TRUE:.*]] : tensor<8x512xi1>
// CHECK:    %[[TMP1:.*]] = arith.andi %[[COND3]], %[[TMP0]] : tensor<8x512xi1>
// CHECK:    %[[MASK3:.*]] = arith.andi %[[TMP1]], %[[MASK]] : tensor<8x512xi1>
// CHECK:    %[[VAL3:.*]] = tt.load %{{.*}}, %[[MASK3]], %{{.*}} : tensor<8x512x!tt.ptr<f32>>

// CHECK:    %[[VAL6:.*]] = arith.select %[[COND2]], %[[VAL5]], %[[VAL3]] : tensor<8x512xi1>, tensor<8x512xf32>

// the fourth load, first use
// condition is mask & ~cond3
//            = mask & (cond3 XOR true)
// CHECK:    %[[TMP0:.*]] = arith.xori %[[COND3]], %[[TRUE:.*]] : tensor<8x512xi1>
// CHECK:    %[[MASK4:.*]] = arith.andi %[[TMP0]], %[[MASK]] : tensor<8x512xi1>
// CHECK:    %[[VAL4:.*]] = tt.load %{{.*}}, %[[MASK4]], %{{.*}} : tensor<8x512x!tt.ptr<f32>>

// the fourth load, second use
// CHECK:    %[[VAL4_2:.*]] = tt.load %{{.*}}, %[[MASK]], %{{.*}} : tensor<8x512x!tt.ptr<f32>>

// CHECK:    %[[VAL7:.*]] = arith.select %[[COND3]], %[[VAL6]], %[[VAL4]] : tensor<8x512xi1>, tensor<8x512xf32>
// CHECK:    tt.store %{{.*}}, %[[VAL7]], %[[MASK]] : tensor<8x512x!tt.ptr<f32>>
// CHECK:    tt.store %{{.*}}, %[[VAL4_2]], %[[MASK]] : tensor<8x512x!tt.ptr<f32>>

module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 32 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 64 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, "ttg.enable-bishengir-simt-optimization" = 10011 : i32, hacc.target = #hacc.target<"Ascend950PR_9589">} {
  tt.func public @triton_unk_fused_cat_0(
      %addr1: tensor<8x512x!tt.ptr<f32>>,
      %addr2: tensor<8x512x!tt.ptr<f32>>,
      %addr3: tensor<8x512x!tt.ptr<f32>>,
      %addr4: tensor<8x512x!tt.ptr<f32>>,
      %mask: tensor<8x512xi1>,
      %cond1: tensor<8x512xi1>,
      %cond2: tensor<8x512xi1>,
      %cond3: tensor<8x512xi1>,
      %addr5: tensor<8x512x!tt.ptr<f32>>,
      %addr6: tensor<8x512x!tt.ptr<f32>>) attributes {noinline = false} {
    %cst0 = arith.constant dense<0.00> : tensor<8x512xf32>
    %val1 = tt.load %addr1, %mask, %cst0 : tensor<8x512x!tt.ptr<f32>>
    %val2 = tt.load %addr2, %mask, %cst0 : tensor<8x512x!tt.ptr<f32>>
    %val3 = tt.load %addr3, %mask, %cst0 : tensor<8x512x!tt.ptr<f32>>
    %val4 = tt.load %addr4, %mask, %cst0 : tensor<8x512x!tt.ptr<f32>>
    %val5 = arith.select %cond1, %val1, %val2 : tensor<8x512xi1>, tensor<8x512xf32>
    %val6 = arith.select %cond2, %val5, %val3 : tensor<8x512xi1>, tensor<8x512xf32>
    %val7 = arith.select %cond3, %val6, %val4 : tensor<8x512xi1>, tensor<8x512xf32>
    tt.store %addr5, %val7, %mask : tensor<8x512x!tt.ptr<f32>>
    tt.store %addr6, %val4, %mask : tensor<8x512x!tt.ptr<f32>>
    tt.return
  }
}