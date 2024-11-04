import torch

bs = 32
seq_len = 128
seq_kv = 127
num_heads = 8
head_dim = 64


query = torch.rand(seq_len, bs, num_heads, head_dim, dtype=torch.float16, device="xpu")
key = torch.rand(seq_kv, bs, num_heads, head_dim, dtype=torch.float16, device="xpu")
value = torch.rand(seq_kv, bs, num_heads, head_dim, dtype=torch.float16, device="xpu")

query.transpose(0, 1)
key.transpose(0, 1)
value.transpose(0, 1)

print('\n\n\n\n\n\n')

with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value)