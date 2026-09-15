// RUN: bishengir-opt --arith-vector-mask-analyze %s | FileCheck %s

// Unrolled binary-search (topk_pack vf_0 shape): each block's midpoint feeds
// three ops in the next block, so the use-def graph has exponentially many
// paths. Without the visited set in analyzeUseAndMark this function takes
// hours (12 blocks ~10^4 revisits per value); with it, milliseconds.

// The single constant_mask is indexed for the masked write.
// CHECK-LABEL: func.func @binary_search_unrolled_vf_0
// CHECK: %[[MASK:.*]] = vector.constant_mask [1] : vector<64xi1>
// CHECK-NEXT: annotation.mark %[[MASK]] {mask_op_idx = 0 : i32} : vector<64xi1>
// The final midpoint select feeding the masked write must carry the
// reached-mask annotation.
// CHECK: annotation.mark %139 {reached_mask_ops_idx = 0 : i32} : vector<64xi32>
// CHECK-NEXT: vector.transfer_write %139, %arg4[%c0], %[[MASK]] {in_bounds = [true]} : vector<64xi32>, memref<1xi32, #hivm.address_space<ub>>
func.func @binary_search_unrolled_vf_0(%arg0: memref<1xi32, #hivm.address_space<ub>>, %arg1: memref<1xi32, #hivm.address_space<ub>>, %arg2: memref<1xi32, #hivm.address_space<ub>>, %arg3: memref<1xi32, #hivm.address_space<ub>>, %arg4: memref<1xi32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %v1 = arith.constant dense<0> : vector<64xi32>
  %v2 = arith.constant dense<-2147483648> : vector<64xi32>
  %v3 = arith.constant dense<1> : vector<64xi32>
  %v4 = arith.constant dense<512> : vector<64xi32>
  %v5 = arith.constant dense<65535> : vector<64xi32>
  %v6 = arith.constant dense<16> : vector<64xi32>
  %v7 = arith.constant 0 : i32
  %v8 = arith.constant 0 : index
  %v9 = vector.constant_mask [1] : vector<64xi1>
  %v10 = vector.transfer_read %arg0[%v8], %v7, %v9 {in_bounds = [true]} : memref<1xi32, #hivm.address_space<ub>>, vector<64xi32>
  %v11 = vector.transfer_read %arg1[%v8], %v7, %v9 {in_bounds = [true]} : memref<1xi32, #hivm.address_space<ub>>, vector<64xi32>
  %v12 = vector.transfer_read %arg2[%v8], %v7, %v9 {in_bounds = [true]} : memref<1xi32, #hivm.address_space<ub>>, vector<64xi32>
  %v13 = vector.transfer_read %arg3[%v8], %v7, %v9 {in_bounds = [true]} : memref<1xi32, #hivm.address_space<ub>>, vector<64xi32>
  %v14 = arith.shli %v11, %v6 : vector<64xi32>
  %v15 = arith.ori %v14, %v5 : vector<64xi32>
  %v16 = arith.cmpi eq, %v10, %v11 : vector<64xi32>
  %v17 = arith.select %v16, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v18 = arith.subi %v13, %v17 : vector<64xi32>
  %v19 = arith.subi %v4, %v18 : vector<64xi32>
  %v20 = arith.subi %v15, %v14 : vector<64xi32>
  %v21 = arith.addi %v20, %v3 : vector<64xi32>
  %v22 = arith.shrsi %v21, %v3 : vector<64xi32>
  %v23 = arith.addi %v14, %v22 : vector<64xi32>
  %v24 = arith.select %v16, %v12, %v2 : vector<64xi1>, vector<64xi32>
  %v25 = arith.cmpi sge, %v24, %v23 : vector<64xi32>
  %v26 = arith.select %v25, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v27 = arith.cmpi sge, %v26, %v19 : vector<64xi32>
  %v28 = arith.subi %v23, %v3 : vector<64xi32>
  %v29 = arith.select %v27, %v15, %v28 : vector<64xi1>, vector<64xi32>
  %v30 = arith.select %v27, %v23, %v14 : vector<64xi1>, vector<64xi32>
  %v31 = arith.subi %v29, %v30 : vector<64xi32>
  %v32 = arith.addi %v31, %v3 : vector<64xi32>
  %v33 = arith.shrsi %v32, %v3 : vector<64xi32>
  %v34 = arith.addi %v30, %v33 : vector<64xi32>
  %v35 = arith.cmpi sge, %v24, %v34 : vector<64xi32>
  %v36 = arith.select %v35, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v37 = arith.cmpi sge, %v36, %v19 : vector<64xi32>
  %v38 = arith.subi %v34, %v3 : vector<64xi32>
  %v39 = arith.select %v37, %v29, %v38 : vector<64xi1>, vector<64xi32>
  %v40 = arith.select %v37, %v34, %v30 : vector<64xi1>, vector<64xi32>
  %v41 = arith.subi %v39, %v40 : vector<64xi32>
  %v42 = arith.addi %v41, %v3 : vector<64xi32>
  %v43 = arith.shrsi %v42, %v3 : vector<64xi32>
  %v44 = arith.addi %v40, %v43 : vector<64xi32>
  %v45 = arith.cmpi sge, %v24, %v44 : vector<64xi32>
  %v46 = arith.select %v45, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v47 = arith.cmpi sge, %v46, %v19 : vector<64xi32>
  %v48 = arith.subi %v44, %v3 : vector<64xi32>
  %v49 = arith.select %v47, %v39, %v48 : vector<64xi1>, vector<64xi32>
  %v50 = arith.select %v47, %v44, %v40 : vector<64xi1>, vector<64xi32>
  %v51 = arith.subi %v49, %v50 : vector<64xi32>
  %v52 = arith.addi %v51, %v3 : vector<64xi32>
  %v53 = arith.shrsi %v52, %v3 : vector<64xi32>
  %v54 = arith.addi %v50, %v53 : vector<64xi32>
  %v55 = arith.cmpi sge, %v24, %v54 : vector<64xi32>
  %v56 = arith.select %v55, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v57 = arith.cmpi sge, %v56, %v19 : vector<64xi32>
  %v58 = arith.subi %v54, %v3 : vector<64xi32>
  %v59 = arith.select %v57, %v49, %v58 : vector<64xi1>, vector<64xi32>
  %v60 = arith.select %v57, %v54, %v50 : vector<64xi1>, vector<64xi32>
  %v61 = arith.subi %v59, %v60 : vector<64xi32>
  %v62 = arith.addi %v61, %v3 : vector<64xi32>
  %v63 = arith.shrsi %v62, %v3 : vector<64xi32>
  %v64 = arith.addi %v60, %v63 : vector<64xi32>
  %v65 = arith.cmpi sge, %v24, %v64 : vector<64xi32>
  %v66 = arith.select %v65, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v67 = arith.cmpi sge, %v66, %v19 : vector<64xi32>
  %v68 = arith.subi %v64, %v3 : vector<64xi32>
  %v69 = arith.select %v67, %v59, %v68 : vector<64xi1>, vector<64xi32>
  %v70 = arith.select %v67, %v64, %v60 : vector<64xi1>, vector<64xi32>
  %v71 = arith.subi %v69, %v70 : vector<64xi32>
  %v72 = arith.addi %v71, %v3 : vector<64xi32>
  %v73 = arith.shrsi %v72, %v3 : vector<64xi32>
  %v74 = arith.addi %v70, %v73 : vector<64xi32>
  %v75 = arith.cmpi sge, %v24, %v74 : vector<64xi32>
  %v76 = arith.select %v75, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v77 = arith.cmpi sge, %v76, %v19 : vector<64xi32>
  %v78 = arith.subi %v74, %v3 : vector<64xi32>
  %v79 = arith.select %v77, %v69, %v78 : vector<64xi1>, vector<64xi32>
  %v80 = arith.select %v77, %v74, %v70 : vector<64xi1>, vector<64xi32>
  %v81 = arith.subi %v79, %v80 : vector<64xi32>
  %v82 = arith.addi %v81, %v3 : vector<64xi32>
  %v83 = arith.shrsi %v82, %v3 : vector<64xi32>
  %v84 = arith.addi %v80, %v83 : vector<64xi32>
  %v85 = arith.cmpi sge, %v24, %v84 : vector<64xi32>
  %v86 = arith.select %v85, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v87 = arith.cmpi sge, %v86, %v19 : vector<64xi32>
  %v88 = arith.subi %v84, %v3 : vector<64xi32>
  %v89 = arith.select %v87, %v79, %v88 : vector<64xi1>, vector<64xi32>
  %v90 = arith.select %v87, %v84, %v80 : vector<64xi1>, vector<64xi32>
  %v91 = arith.subi %v89, %v90 : vector<64xi32>
  %v92 = arith.addi %v91, %v3 : vector<64xi32>
  %v93 = arith.shrsi %v92, %v3 : vector<64xi32>
  %v94 = arith.addi %v90, %v93 : vector<64xi32>
  %v95 = arith.cmpi sge, %v24, %v94 : vector<64xi32>
  %v96 = arith.select %v95, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v97 = arith.cmpi sge, %v96, %v19 : vector<64xi32>
  %v98 = arith.subi %v94, %v3 : vector<64xi32>
  %v99 = arith.select %v97, %v89, %v98 : vector<64xi1>, vector<64xi32>
  %v100 = arith.select %v97, %v94, %v90 : vector<64xi1>, vector<64xi32>
  %v101 = arith.subi %v99, %v100 : vector<64xi32>
  %v102 = arith.addi %v101, %v3 : vector<64xi32>
  %v103 = arith.shrsi %v102, %v3 : vector<64xi32>
  %v104 = arith.addi %v100, %v103 : vector<64xi32>
  %v105 = arith.cmpi sge, %v24, %v104 : vector<64xi32>
  %v106 = arith.select %v105, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v107 = arith.cmpi sge, %v106, %v19 : vector<64xi32>
  %v108 = arith.subi %v104, %v3 : vector<64xi32>
  %v109 = arith.select %v107, %v99, %v108 : vector<64xi1>, vector<64xi32>
  %v110 = arith.select %v107, %v104, %v100 : vector<64xi1>, vector<64xi32>
  %v111 = arith.subi %v109, %v110 : vector<64xi32>
  %v112 = arith.addi %v111, %v3 : vector<64xi32>
  %v113 = arith.shrsi %v112, %v3 : vector<64xi32>
  %v114 = arith.addi %v110, %v113 : vector<64xi32>
  %v115 = arith.cmpi sge, %v24, %v114 : vector<64xi32>
  %v116 = arith.select %v115, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v117 = arith.cmpi sge, %v116, %v19 : vector<64xi32>
  %v118 = arith.subi %v114, %v3 : vector<64xi32>
  %v119 = arith.select %v117, %v109, %v118 : vector<64xi1>, vector<64xi32>
  %v120 = arith.select %v117, %v114, %v110 : vector<64xi1>, vector<64xi32>
  %v121 = arith.subi %v119, %v120 : vector<64xi32>
  %v122 = arith.addi %v121, %v3 : vector<64xi32>
  %v123 = arith.shrsi %v122, %v3 : vector<64xi32>
  %v124 = arith.addi %v120, %v123 : vector<64xi32>
  %v125 = arith.cmpi sge, %v24, %v124 : vector<64xi32>
  %v126 = arith.select %v125, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v127 = arith.cmpi sge, %v126, %v19 : vector<64xi32>
  %v128 = arith.subi %v124, %v3 : vector<64xi32>
  %v129 = arith.select %v127, %v119, %v128 : vector<64xi1>, vector<64xi32>
  %v130 = arith.select %v127, %v124, %v120 : vector<64xi1>, vector<64xi32>
  %v131 = arith.subi %v129, %v130 : vector<64xi32>
  %v132 = arith.addi %v131, %v3 : vector<64xi32>
  %v133 = arith.shrsi %v132, %v3 : vector<64xi32>
  %v134 = arith.addi %v130, %v133 : vector<64xi32>
  %v135 = arith.cmpi sge, %v24, %v134 : vector<64xi32>
  %v136 = arith.select %v135, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v137 = arith.cmpi sge, %v136, %v19 : vector<64xi32>
  %v138 = arith.subi %v134, %v3 : vector<64xi32>
  %v139 = arith.select %v137, %v129, %v138 : vector<64xi1>, vector<64xi32>
  %v140 = arith.select %v137, %v134, %v130 : vector<64xi1>, vector<64xi32>
  %v141 = arith.subi %v139, %v140 : vector<64xi32>
  %v142 = arith.addi %v141, %v3 : vector<64xi32>
  %v143 = arith.shrsi %v142, %v3 : vector<64xi32>
  %v144 = arith.addi %v140, %v143 : vector<64xi32>
  %v145 = arith.cmpi sge, %v24, %v144 : vector<64xi32>
  %v146 = arith.select %v145, %v3, %v1 : vector<64xi1>, vector<64xi32>
  %v147 = arith.cmpi sge, %v146, %v19 : vector<64xi32>
  %v148 = arith.select %v147, %v144, %v140 : vector<64xi1>, vector<64xi32>
  vector.transfer_write %v148, %arg4[%v8], %v9 {in_bounds = [true]} : vector<64xi32>, memref<1xi32, #hivm.address_space<ub>>
  return
}
