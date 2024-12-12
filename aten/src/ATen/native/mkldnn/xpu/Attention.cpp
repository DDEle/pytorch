#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
#include <torch/library.h>

namespace {

// This function is used to produce an attn_mask
// in a standard format that can be consumed by both
// the math and memory efficient attn_mask implementation
//  Args:
//    attn_mask: attn_mask of shape (B, L, S) or (L, S) or (B, N_heads, L, S)
at::Tensor convert_boolean_attn_mask(
    const at::Tensor& attn_mask,
    at::TensorOptions& opt) {
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask.dtype() == at::kBool) {
    return at::where(
        attn_mask.logical_not(),
        -std::numeric_limits<double>::infinity(),
        at::scalar_tensor(0.0, opt));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

bool check_head_dim_size_xpu(sdp::sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if ((query_size_last != key_size_last) ||
      (query_size_last != value_size_last)) {
    if (debug) {
      TORCH_WARN(
          "OneDNN Graph's attention requires q,k,v to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }
  if (query_size_last > 256) {
    if (debug) {
      TORCH_WARN(
          "OneDNN Graph's attention requires q,k,v to have head dimension less than 256.",
          " Got ",
          query_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_no_grad(sdp::sdp_params const& params, bool debug) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if (debug && any_inputs_require_grad && gradmode_enabled) {
    TORCH_WARN("Backward or grad to be supported.");
  }
  return !any_inputs_require_grad || !gradmode_enabled;
}

bool use_overrideable_xpu(sdp::sdp_params const& params, bool debug) {
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(
      sdp::sdp_params const&, bool)>(
      sdp::check_nested_tensor,
      sdp::check_for_dropout,
      sdp::check_tensor_shapes,
      sdp::check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      sdp::check_attn_mask_shape,
      sdp::check_nonzero_sequence_lengths_dense,
      sdp::check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_head_dim_size_xpu,
      check_no_grad);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return sdp::check_tensor_dtype(params, supported_dtypes, debug);
}

sdp::SDPBackend select_sdp_backend_xpu(sdp::sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();
  // use overrideable linked to onednn graph as mem efficient implementation
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP()) {
    return sdp::SDPBackend::error;
  }

  // Get ideal kernel ordering
  const std::array<sdp::SDPBackend, 2> priority_order{
      sdp::SDPBackend::overrideable,
      sdp::SDPBackend::math,
  };

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : priority_order) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            use_overrideable_xpu(kernel_params, print_debug)) {
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_overrideable_xpu did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("OneDNN Graph kernel not used because:");
  use_overrideable_xpu(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return sdp::SDPBackend::error;
}
} // namespace

namespace at::native {
int64_t _fused_sdp_choice_xpu(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  sdp::sdp_params kernel_params{
      query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = select_sdp_backend_xpu(kernel_params);

  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
_scaled_dot_product_fused_attention_overrideable_xpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_xpu: Accept only 4 dims inputs shape of {(B), H, T, K}");
  TORCH_CHECK(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_fused_attention_overrideable_xpu: K/V should have the same batch / seq / num_head");
  TORCH_CHECK(
      dropout_p == 0.0,
      "scaled_dot_product_fused_attention_overrideable_xpu: Currently do not support dropout > 0");
  TORCH_CHECK(
      !(attn_bias.has_value() && is_causal),
      "scaled_dot_product_fused_attention_overrideable_xpu: attn_bias cannot present with is_causal");

  const int64_t batch_size = query.size(0);
  const int64_t num_head = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t head_dim = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);

  auto opts = query.options();
  auto output = at::empty({batch_size, num_head, seq_len_q, head_dim}, opts);
  // auto logsumexp =
  //     at::empty({batch_size, num_head, seq_len_q}, opts.dtype(at::kFloat));
  auto logsumexp = at::empty({}, opts.dtype(at::kFloat));

  std::optional<at::Tensor> attn_mask_fallback;
  if (attn_bias.has_value()) {
    attn_mask_fallback = attn_bias;
  } else {
    if (is_causal) {
      auto attn_mask_fallback =
          at::ones_symint({seq_len_q, seq_len_kv}, opts.dtype(at::kBool))
              .tril();
      attn_mask_fallback = convert_boolean_attn_mask(attn_mask_fallback, opts);
    } else {
      attn_mask_fallback = std::nullopt;
    }
  }
  at::native::onednn::graph::gpu_float_sdpa(
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head,
      num_head_kv,
      head_dim,
      head_dim_v,
      query,
      key,
      value,
      attn_mask_fallback,
      false, // is_causal fallback with attn_mask for now
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim)),
      output);

  // rng and debug mask not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  auto debug_attn_mask = at::empty(
      {batch_size, num_head, seq_len_q, seq_len_kv}, at::dtype(at::kFloat));

  return std::make_tuple(
      output,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native
