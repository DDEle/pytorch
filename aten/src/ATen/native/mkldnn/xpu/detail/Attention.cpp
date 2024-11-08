#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Graph.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

#include <omp.h>
#include <oneapi/dnnl/dnnl.hpp>

using namespace at::native::onednn::graph;

namespace {
void allocate_sycl_graph_mem(
    std::vector<dnnl::graph::tensor>& tensors,
    const logical_tensor& lt,
    const engine& eng,
    const at::Tensor& input) {
  dnnl::graph::tensor new_ts{lt, eng, input.data_ptr()};
  tensors.push_back(new_ts);
}

partition create_sdpa_graph_partition(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int size_per_head,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_mask,
    const at::Tensor& output,
    data_type dtype) {
  // graph building and partitioning
  // currently, we assume that Q and K have same sequence length

  dims q_input_shape = {batch_size, num_head, seq_len_q, size_per_head};
  dims kv_input_shape = {batch_size, num_head, seq_len_k, size_per_head};
  dims qk_output_shape = {batch_size, num_head, seq_len_q, seq_len_k};
  dims scale_shape = {1};
  dims attention_mask_shape = {attn_mask.sizes().vec()};
  size_t lt_id = 0;

  logical_tensor query_input{
      lt_id++, dtype, q_input_shape, query.strides().vec()};
  logical_tensor key_input{lt_id++, dtype, kv_input_shape, key.strides().vec()};

  logical_tensor matmul_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op matmul_qk{
      0,
      op::kind::MatMul,
      {query_input, key_input},
      {matmul_qk_out},
      "matmul_qk"};
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  logical_tensor scale_factor{
      lt_id++,
      dtype,
      scale_shape,
      logical_tensor::layout_type::strided,
      logical_tensor::property_type::constant};
  logical_tensor scaled_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op scale_div{
      1,
      op::kind::Divide,
      {matmul_qk_out, scale_factor},
      {scaled_qk_out},
      "scale_div"};

  logical_tensor attention_mask{
      lt_id++, dtype, attention_mask_shape, attn_mask.strides().vec()};
  logical_tensor masked_qk_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  op mask_add{
      2,
      op::kind::Add,
      {scaled_qk_out, attention_mask},
      {masked_qk_out},
      "mask_add"};

  op softmax{3, op::kind::SoftMax, "softmax"};
  softmax.set_attr<int64_t>(op::attr::axis, -1);

  logical_tensor softmax_out{
      lt_id++, dtype, qk_output_shape, logical_tensor::layout_type::strided};
  softmax.add_input(masked_qk_out);
  softmax.add_output(softmax_out);

  logical_tensor value_input{
      lt_id++, dtype, kv_input_shape, value.strides().vec()};
  logical_tensor matmul_v_out{
      lt_id++, dtype, q_input_shape, output.strides().vec()};

  op matmul_v{
      4,
      op::kind::MatMul,
      {softmax_out, value_input},
      {matmul_v_out},
      "matmul_v"};

  engine::kind ekind = engine::kind::gpu;
  graph g(ekind);
  g.add_op(matmul_qk);
  g.add_op(scale_div);
  g.add_op(mask_add);
  g.add_op(softmax);
  g.add_op(matmul_v);
  g.finalize();
  auto partitions = g.get_partitions();
  TORCH_CHECK(
      (partitions.size() == 1) && partitions[0].is_supported(),
      "oneDNN Graph doesn't support this fusion pattern. If you'd like its support, please submit a ticket.");
  return partitions[0];
}
} // namespace

namespace at::native::onednn::graph {

// // Thread local data-structures are required if multiple thread-pools
// // of a PyTorch process would be used for inference.
// thread_local std::unordered_map<std::bitset<32>, dnnl::graph::partition>
//     partition_map_;
// // Compiled partition (fused kernel) cache
// // Adopted from
// // https://github.com/lamerman/cpp-lru-cache/blob/master/include/lrucache.hpp

// thread_local std::list<key_value_pair_t> cache_items_list_;
// thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
//     fused_kernel_cache_map_;
// // cache capacity is arbitrary
// // TODO: Add an API to manipulate cache capacity
// thread_local size_t capacity_ = 1024;

TORCH_API void gpu_float_sdpa(
    int batch_size,
    int seq_len_q,
    int seq_len_k,
    int num_head,
    int size_per_head,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    const float& softmax_scale,
    const Tensor& output) {
  auto eng = GpuEngineManager::Instance().get_engine(
      {c10::kXPU, c10::xpu::current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  Tensor softmax_scale1 = at::full(
      {},
      1 / softmax_scale,
      TensorOptions().dtype(c10::ScalarType::Half).device(DeviceType::XPU));

  const data_type logical_tensor_dtype =
      query.scalar_type() == c10::ScalarType::Float      ? data_type::f32
      : query.scalar_type() == c10::ScalarType::Half     ? data_type::f16
      : query.scalar_type() == c10::ScalarType::BFloat16 ? data_type::bf16
                                                         : data_type::undef;
  TORCH_CHECK(
      (logical_tensor_dtype != data_type::undef),
      "Only FP16 & BF16 & FP32 datatypes are currently supported");

  // cache key creation
  // patternID is determined on the basis of the arguments provided
  std::bitset<32> patternID;
  if (logical_tensor_dtype == data_type::f32) {
    // bit 3 corresponds to float32 dtype
    patternID.set(3, 1);
  } else {
    // bit 2 corresponds to float16 dtype
    patternID.set(2, 1);
  }
  // sdp pattern
  patternID.set(4, 1);
  // Refer to comments in Graph.cpp. The first 8 bits are reserved
  int pos = 8;
  // add more bool configs

  // first check cache
  // The key has a pattern ID, as well as the shapes of input tenors
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  // We use this because different thread-pools may be used
  map_key.push_back(omp_get_max_threads());
  map_key.push_back(static_cast<int64_t>(patternID.to_ullong()));
  map_key.insert(map_key.end(), key.sizes().begin(), key.sizes().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.sizes().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.sizes().end());
  map_key.insert(
      map_key.end(),
      softmax_scale1.sizes().begin(),
      softmax_scale1.sizes().end());
  auto iter = cache_lookup(map_key);

  cp_entry cp;
  if (iter == cache_end()) {
    auto graph_partition_iter = partition_map_lookup(patternID);
    if (graph_partition_iter == partition_map_end()) {
      // partition cache no hit
      TORCH_CHECK(
          ((logical_tensor_dtype == data_type::f16) ||
           (logical_tensor_dtype == data_type::f32)),
          "Only F16 & FP32 datatypes are currently supported");
      // graph building and partitioning
      partition sdp_partition = create_sdpa_graph_partition(
          batch_size,
          seq_len_q,
          seq_len_k,
          num_head,
          size_per_head,
          query,
          key,
          value,
          attn_mask,
          output,
          logical_tensor_dtype);

      insert_in_partition_cache(patternID, sdp_partition);
      graph_partition_iter = partition_map_lookup(patternID);
    }

    cp.partition_ = graph_partition_iter->second;
    // partition compilation
    compile_partition(cp, eng);
  } else {
    cp = iter->second->second;
  }

  // partition execution
  auto& inputs = cp.inputLogicalTensors_;
  auto& outputs = cp.outputLogicalTensors_;
  cp.inputLLGATensors_.clear();
  cp.outputLLGATensors_.clear();
  allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[0], eng, query);
  allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[1], eng, key);
  allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[2], eng, softmax_scale1);
  if (1) {
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[3], eng, attn_mask);
    allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[4], eng, value);
  }
  allocate_sycl_graph_mem(cp.inputLLGATensors_, inputs[3], eng, value);
  allocate_sycl_graph_mem(cp.outputLLGATensors_, outputs[0], eng, output);
  cp.cp_.execute(strm, cp.inputLLGATensors_, cp.outputLLGATensors_);

  if (iter == cache_end()) {
    // cache the compiled kernel
    insert_in_fused_kernel_cache(map_key, cp);
  }
}
} // namespace at::native::onednn::graph
