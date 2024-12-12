

// WARNING: THIS FILE IS AUTOGENERATED BY torchgen. DO NOT MODIFY BY HAND.
// See https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436 for details

#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__adaptive_avg_pool2d(AtenTensorHandle self, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__adaptive_avg_pool2d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__adaptive_avg_pool3d(AtenTensorHandle self, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__adaptive_avg_pool3d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__addmm_activation(AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha, int32_t use_gelu, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__cdist_backward(AtenTensorHandle grad, AtenTensorHandle x1, AtenTensorHandle x2, double p, AtenTensorHandle cdist, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__cdist_forward(AtenTensorHandle x1, AtenTensorHandle x2, double p, int64_t* compute_mode, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__dyn_quant_matmul_4bit(AtenTensorHandle inp, AtenTensorHandle packed_weights, int64_t block_size, int64_t in_features, int64_t out_features, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__dyn_quant_pack_4bit_weight(AtenTensorHandle weights, AtenTensorHandle scales_zeros, AtenTensorHandle* bias, int64_t block_size, int64_t in_features, int64_t out_features, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__efficientzerotensor(const int64_t* size, int64_t size_len_, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__embedding_bag(AtenTensorHandle weight, AtenTensorHandle indices, AtenTensorHandle offsets, int32_t scale_grad_by_freq, int64_t mode, int32_t sparse, AtenTensorHandle* per_sample_weights, int32_t include_last_offset, int64_t padding_idx, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2, AtenTensorHandle* ret3);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__embedding_bag_dense_backward(AtenTensorHandle grad, AtenTensorHandle indices, AtenTensorHandle offset2bag, AtenTensorHandle bag_size, AtenTensorHandle maximum_indices, int64_t num_weights, int32_t scale_grad_by_freq, int64_t mode, AtenTensorHandle* per_sample_weights, int64_t padding_idx, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__embedding_bag_forward_only(AtenTensorHandle weight, AtenTensorHandle indices, AtenTensorHandle offsets, int32_t scale_grad_by_freq, int64_t mode, int32_t sparse, AtenTensorHandle* per_sample_weights, int32_t include_last_offset, int64_t padding_idx, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2, AtenTensorHandle* ret3);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__embedding_bag_per_sample_weights_backward(AtenTensorHandle grad, AtenTensorHandle weight, AtenTensorHandle indices, AtenTensorHandle offsets, AtenTensorHandle offset2bag, int64_t mode, int64_t padding_idx, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__fft_c2c(AtenTensorHandle self, const int64_t* dim, int64_t dim_len_, int64_t normalization, int32_t forward, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__fft_r2c(AtenTensorHandle self, const int64_t* dim, int64_t dim_len_, int64_t normalization, int32_t onesided, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__fused_moving_avg_obs_fq_helper(AtenTensorHandle self, AtenTensorHandle observer_on, AtenTensorHandle fake_quant_on, AtenTensorHandle running_min, AtenTensorHandle running_max, AtenTensorHandle scale, AtenTensorHandle zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, int32_t per_row_fake_quant, int32_t symmetric_quant, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__fused_moving_avg_obs_fq_helper_functional(AtenTensorHandle self, AtenTensorHandle observer_on, AtenTensorHandle fake_quant_on, AtenTensorHandle running_min, AtenTensorHandle running_max, AtenTensorHandle scale, AtenTensorHandle zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, int32_t per_row_fake_quant, int32_t symmetric_quant, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2, AtenTensorHandle* ret3, AtenTensorHandle* ret4, AtenTensorHandle* ret5);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__histogramdd_from_bin_cts(AtenTensorHandle self, const int64_t* bins, int64_t bins_len_, const double** range, int64_t range_len_, AtenTensorHandle* weight, int32_t density, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__int_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__pdist_backward(AtenTensorHandle grad, AtenTensorHandle self, double p, AtenTensorHandle pdist, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__pdist_forward(AtenTensorHandle self, double p, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__scaled_dot_product_flash_attention_for_cpu(AtenTensorHandle query, AtenTensorHandle key, AtenTensorHandle value, double dropout_p, int32_t is_causal, AtenTensorHandle* attn_mask, double* scale, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__scaled_dot_product_flash_attention_for_cpu_backward(AtenTensorHandle grad_out, AtenTensorHandle query, AtenTensorHandle key, AtenTensorHandle value, AtenTensorHandle out, AtenTensorHandle logsumexp, double dropout_p, int32_t is_causal, AtenTensorHandle* attn_mask, double* scale, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__scaled_dot_product_fused_attention_overrideable(AtenTensorHandle query, AtenTensorHandle key, AtenTensorHandle value, AtenTensorHandle* attn_bias, double dropout_p, int32_t is_causal, int32_t return_debug_mask, double* scale, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2, AtenTensorHandle* ret3, int64_t* ret4, int64_t* ret5, AtenTensorHandle* ret6, AtenTensorHandle* ret7, AtenTensorHandle* ret8);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__scaled_dot_product_fused_attention_overrideable_backward(AtenTensorHandle grad_out, AtenTensorHandle query, AtenTensorHandle key, AtenTensorHandle value, AtenTensorHandle attn_bias, const int32_t* grad_input_mask, int64_t grad_input_mask_len_, AtenTensorHandle out, AtenTensorHandle logsumexp, AtenTensorHandle cum_seq_q, AtenTensorHandle cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, int32_t is_causal, AtenTensorHandle philox_seed, AtenTensorHandle philox_offset, double* scale, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2, AtenTensorHandle* ret3);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__segment_reduce_backward(AtenTensorHandle grad, AtenTensorHandle output, AtenTensorHandle data, const char* reduce, AtenTensorHandle* lengths, AtenTensorHandle* offsets, int64_t axis, double* initial, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__to_sparse(AtenTensorHandle self, int32_t* layout, const int64_t** blocksize, int64_t blocksize_len_, int64_t* dense_dim, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__trilinear(AtenTensorHandle i1, AtenTensorHandle i2, AtenTensorHandle i3, const int64_t* expand1, int64_t expand1_len_, const int64_t* expand2, int64_t expand2_len_, const int64_t* expand3, int64_t expand3_len_, const int64_t* sumdim, int64_t sumdim_len_, int64_t unroll_dim, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__weight_int8pack_mm(AtenTensorHandle self, AtenTensorHandle mat2, AtenTensorHandle scales, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_adaptive_max_pool2d(AtenTensorHandle self, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_adaptive_max_pool2d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_adaptive_max_pool3d(AtenTensorHandle self, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_adaptive_max_pool3d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_add_Scalar(AtenTensorHandle self, double other, double alpha, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_add_Tensor(AtenTensorHandle self, AtenTensorHandle other, double alpha, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_addbmm(AtenTensorHandle self, AtenTensorHandle batch1, AtenTensorHandle batch2, double beta, double alpha, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_addmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat1, AtenTensorHandle mat2, double beta, double alpha);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_addmv(AtenTensorHandle self, AtenTensorHandle mat, AtenTensorHandle vec, double beta, double alpha, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_angle(AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_avg_pool2d(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, int32_t ceil_mode, int32_t count_include_pad, int64_t* divisor_override, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_avg_pool2d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, int32_t ceil_mode, int32_t count_include_pad, int64_t* divisor_override, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_avg_pool3d(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, int32_t ceil_mode, int32_t count_include_pad, int64_t* divisor_override, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_avg_pool3d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, int32_t ceil_mode, int32_t count_include_pad, int64_t* divisor_override, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_baddbmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle batch1, AtenTensorHandle batch2, double beta, double alpha);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_bernoulli__Tensor(AtenTensorHandle self, AtenTensorHandle p, AtenGeneratorHandle* generator);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_bernoulli__float(AtenTensorHandle self, double p, AtenGeneratorHandle* generator);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_bmm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_bucketize_Tensor(AtenTensorHandle self, AtenTensorHandle boundaries, int32_t out_int32, int32_t right, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cat(const AtenTensorHandle* tensors, int64_t tensors_len_, int64_t dim, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cholesky_inverse(AtenTensorHandle self, int32_t upper, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cholesky_solve(AtenTensorHandle self, AtenTensorHandle input2, int32_t upper, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_convolution(AtenTensorHandle input, AtenTensorHandle weight, AtenTensorHandle* bias, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t transposed, const int64_t* output_padding, int64_t output_padding_len_, int64_t groups, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_convolution_backward(AtenTensorHandle grad_output, AtenTensorHandle input, AtenTensorHandle weight, const int64_t** bias_sizes, int64_t bias_sizes_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t transposed, const int64_t* output_padding, int64_t output_padding_len_, int64_t groups, const int32_t* output_mask, int64_t output_mask_len_, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cummax(AtenTensorHandle self, int64_t dim, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cummin(AtenTensorHandle self, int64_t dim, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cumprod(AtenTensorHandle self, int64_t dim, int32_t* dtype, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_cumsum(AtenTensorHandle self, int64_t dim, int32_t* dtype, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_exponential(AtenTensorHandle self, double lambd, AtenGeneratorHandle* generator, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_fractional_max_pool2d(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle random_samples, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_fractional_max_pool2d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_fractional_max_pool3d(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle random_samples, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_fractional_max_pool3d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_gcd(AtenTensorHandle self, AtenTensorHandle other, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_geqrf(AtenTensorHandle self, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_grid_sampler_2d_backward(AtenTensorHandle grad_output, AtenTensorHandle input, AtenTensorHandle grid, int64_t interpolation_mode, int64_t padding_mode, int32_t align_corners, const int32_t* output_mask, int64_t output_mask_len_, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_histc(AtenTensorHandle self, int64_t bins, double min, double max, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_histogram_bin_ct(AtenTensorHandle self, int64_t bins, const double** range, int64_t range_len_, AtenTensorHandle* weight, int32_t density, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_index_Tensor(AtenTensorHandle self, const AtenTensorHandle** indices, int64_t indices_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_index_put(AtenTensorHandle self, const AtenTensorHandle** indices, int64_t indices_len_, AtenTensorHandle values, int32_t accumulate, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_index_reduce(AtenTensorHandle self, int64_t dim, AtenTensorHandle index, AtenTensorHandle source, const char* reduce, int32_t include_self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_kthvalue(AtenTensorHandle self, int64_t k, int64_t dim, int32_t keepdim, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_logcumsumexp(AtenTensorHandle self, int64_t dim, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_lu_unpack(AtenTensorHandle LU_data, AtenTensorHandle LU_pivots, int32_t unpack_data, int32_t unpack_pivots, AtenTensorHandle* ret0, AtenTensorHandle* ret1, AtenTensorHandle* ret2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_masked_scatter(AtenTensorHandle self, AtenTensorHandle mask, AtenTensorHandle source, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_masked_scatter_backward(AtenTensorHandle grad_output, AtenTensorHandle mask, const int64_t* sizes, int64_t sizes_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_masked_select(AtenTensorHandle self, AtenTensorHandle mask, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_pool2d_with_indices(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t ceil_mode, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_pool2d_with_indices_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t ceil_mode, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_pool3d_with_indices(AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t ceil_mode, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_pool3d_with_indices_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* kernel_size, int64_t kernel_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, const int64_t* dilation, int64_t dilation_len_, int32_t ceil_mode, AtenTensorHandle indices, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_unpool2d(AtenTensorHandle self, AtenTensorHandle indices, const int64_t* output_size, int64_t output_size_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_max_unpool3d(AtenTensorHandle self, AtenTensorHandle indices, const int64_t* output_size, int64_t output_size_len_, const int64_t* stride, int64_t stride_len_, const int64_t* padding, int64_t padding_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_median(AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mm_out(AtenTensorHandle out, AtenTensorHandle self, AtenTensorHandle mat2);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mode(AtenTensorHandle self, int64_t dim, int32_t keepdim, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mul_Scalar(AtenTensorHandle self, double other, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mul_Tensor(AtenTensorHandle self, AtenTensorHandle other, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_nanmedian(AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_native_dropout(AtenTensorHandle input, double p, int32_t* train, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_nonzero(AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_normal_functional(AtenTensorHandle self, double mean, double std, AtenGeneratorHandle* generator, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_ormqr(AtenTensorHandle self, AtenTensorHandle input2, AtenTensorHandle input3, int32_t left, int32_t transpose, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_polar(AtenTensorHandle abs, AtenTensorHandle angle, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_pow_Scalar(double self, AtenTensorHandle exponent, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_pow_Tensor_Scalar(AtenTensorHandle self, double exponent, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_pow_Tensor_Tensor(AtenTensorHandle self, AtenTensorHandle exponent, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_rand(const int64_t* size, int64_t size_len_, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_rand_generator(const int64_t* size, int64_t size_len_, AtenGeneratorHandle* generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randint(int64_t high, const int64_t* size, int64_t size_len_, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randint_generator(int64_t high, const int64_t* size, int64_t size_len_, AtenGeneratorHandle* generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randint_low(int64_t low, int64_t high, const int64_t* size, int64_t size_len_, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randint_low_out(AtenTensorHandle out, int64_t low, int64_t high, const int64_t* size, int64_t size_len_);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randn(const int64_t* size, int64_t size_len_, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randn_generator(const int64_t* size, int64_t size_len_, AtenGeneratorHandle* generator, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_randperm(int64_t n, int32_t* dtype, int32_t* layout, int32_t* device, int32_t device_index_, int32_t* pin_memory, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_repeat_interleave_Tensor(AtenTensorHandle repeats, int64_t* output_size, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_replication_pad1d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* padding, int64_t padding_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_replication_pad2d_backward(AtenTensorHandle grad_output, AtenTensorHandle self, const int64_t* padding, int64_t padding_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_reshape(AtenTensorHandle self, const int64_t* shape, int64_t shape_len_, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_resize_(AtenTensorHandle self, const int64_t* size, int64_t size_len_, int32_t* memory_format);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_resize_as_(AtenTensorHandle self, AtenTensorHandle the_template, int32_t* memory_format);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_scatter_src_out(AtenTensorHandle out, AtenTensorHandle self, int64_t dim, AtenTensorHandle index, AtenTensorHandle src);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_scatter_value_out(AtenTensorHandle out, AtenTensorHandle self, int64_t dim, AtenTensorHandle index, double value);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_scatter_reduce_two_out(AtenTensorHandle out, AtenTensorHandle self, int64_t dim, AtenTensorHandle index, AtenTensorHandle src, const char* reduce, int32_t include_self);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_searchsorted_Scalar(AtenTensorHandle sorted_sequence, double self, int32_t out_int32, int32_t right, const char** side, AtenTensorHandle* sorter, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_searchsorted_Tensor(AtenTensorHandle sorted_sequence, AtenTensorHandle self, int32_t out_int32, int32_t right, const char** side, AtenTensorHandle* sorter, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_segment_reduce(AtenTensorHandle data, const char* reduce, AtenTensorHandle* lengths, AtenTensorHandle* indices, AtenTensorHandle* offsets, int64_t axis, int32_t unsafe, double* initial, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_slice_Tensor(AtenTensorHandle self, int64_t dim, int64_t* start, int64_t* end, int64_t step, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_soft_margin_loss_backward(AtenTensorHandle grad_output, AtenTensorHandle self, AtenTensorHandle target, int64_t reduction, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_sort(AtenTensorHandle self, int64_t dim, int32_t descending, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_sort_stable(AtenTensorHandle self, int32_t* stable, int64_t dim, int32_t descending, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_to_sparse(AtenTensorHandle self, int32_t* layout, const int64_t** blocksize, int64_t blocksize_len_, int64_t* dense_dim, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_topk(AtenTensorHandle self, int64_t k, int64_t dim, int32_t largest, int32_t sorted, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_triangular_solve(AtenTensorHandle self, AtenTensorHandle A, int32_t upper, int32_t transpose, int32_t unitriangular, AtenTensorHandle* ret0, AtenTensorHandle* ret1);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_uniform(AtenTensorHandle self, double from, double to, AtenGeneratorHandle* generator, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_upsample_bicubic2d_backward(AtenTensorHandle grad_output, const int64_t* output_size, int64_t output_size_len_, const int64_t* input_size, int64_t input_size_len_, int32_t align_corners, double* scales_h, double* scales_w, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_upsample_linear1d_backward(AtenTensorHandle grad_output, const int64_t* output_size, int64_t output_size_len_, const int64_t* input_size, int64_t input_size_len_, int32_t align_corners, double* scales, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_upsample_trilinear3d_backward(AtenTensorHandle grad_output, const int64_t* output_size, int64_t output_size_len_, const int64_t* input_size, int64_t input_size_len_, int32_t align_corners, double* scales_d, double* scales_h, double* scales_w, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_view_dtype(AtenTensorHandle self, int32_t dtype, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_view_as_complex(AtenTensorHandle self, AtenTensorHandle* ret0);
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_view_as_real(AtenTensorHandle self, AtenTensorHandle* ret0);

#ifdef __cplusplus
} // extern "C"
#endif
