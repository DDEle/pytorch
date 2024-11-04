// #include <ATen/native/transformers/xpu/sdp_utils.h>
// #include <ATen/native/transformers/sdp_utils_cpp.h>


// namespace sdp {
// namespace {
// std::array<SDPBackend, num_backends> priority_order(sdp_params const& params) {
//   constexpr std::array<SDPBackend, num_backends> default_order{
//       SDPBackend::efficient_attention,
//       SDPBackend::math};

//   return default_order;
// }
// }

// SDPBackend select_sdp_backend_xpu(sdp_params const& kernel_params) {
//   // This function defines the priority order of the different sdp backends
//   // 1. Flash Attention
//   // 2. Math fallback
//   auto& ctx = at::globalContext();
//   if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP()) {
//     return SDPBackend::error;
//   }
//   // Get ideal kernel ordering
//   const auto ordering = priority_order(kernel_params);

//   // Because TORCHCHECK checks if condition is true we negate debug so that
//   // The statements will be printed when debug is true
//   bool print_debug = false;
//   for (auto& backend : ordering) {
//     switch (backend) {
//       case SDPBackend::flash_attention:
//         if (use_mem_efficient_attention_xpu(kernel_params, print_debug)) {
//           return SDPBackend::flash_attention;
//         }
//         break;
//       case SDPBackend::math:
//         if (ctx.userEnabledMathSDP()) {
//           return SDPBackend::math;
//         }
//         break;
//       default:
//         TORCH_CHECK(false, "Invalid backend");
//     }
//   }
//   // If we have gotten to this point then two things have happened:
//   // 1. use_mem_efficient_attention did not satisfy the
//   // constraints to be ran
//   // 2. The user has explicitly disabled the math kernel
//   // We then re-run the kernel checks with debug enabled to print out the
//   // reason why the kernel was not selected

//   print_debug = true;
//   TORCH_WARN("Memory efficient kernel not used because:");
//   use_mem_efficient_attention_xpu(kernel_params, print_debug);
//   TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
//   return SDPBackend::error;
// }


// bool use_mem_efficient_attention_xpu(sdp_params const& params, bool debug) {
//   constexpr auto xpu_supported_mem_efficient_dtypes =
//       array_of<at::ScalarType>(at::kFloat, at::kDouble, at::kBFloat16, at::kHalf);

//   // Define gate functions that determine if a mem_efficient kernel can be run
//   constexpr auto constraints = array_of<bool (*)(sdp_params const&, bool)>(
//       check_runtime_disabled_mem_efficient,
//     //   check_nested_tensor,
//     //   check_for_dropout,
//     //   check_tensor_shapes,
//     //   check_batch_size_and_num_heads_dense<false /*supports_grouped_query_attention*/>,
//     //   check_attn_mask_shape,
//     //   check_head_dim_size_xpu,
//     //   check_nonzero_sequence_lengths_dense,
//     //   check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>
//       );
//   for (auto& constraint : constraints) {
//     if (!constraint(params, debug)) {
//       return false;
//     }
//   }

//   return check_tensor_dtype(params, xpu_supported_mem_efficient_dtypes, debug);
// }

// }