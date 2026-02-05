import torch


@torch.no_grad()
def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if reorder_channel == "in":
        weight1 = linear_module.weight.data[:, full_attn_mask]
        weight2 = linear_module.weight.data[:, ~full_attn_mask]
        reordered_weight = torch.cat([weight1, weight2], dim=1)
    else:
        weight1 = linear_module.weight.data[full_attn_mask, :]
        weight2 = linear_module.weight.data[~full_attn_mask, :]
        reordered_weight = torch.cat([weight1, weight2], dim=0)
    linear_module.weight.data = reordered_weight
    # for linear modules with bias
    if linear_module.bias is not None:
        bias1 = linear_module.bias.data[full_attn_mask]
        bias2 = linear_module.bias.data[~full_attn_mask]
        reordered_bias = torch.cat([bias1, bias2], dim=0)
        linear_module.bias.data = reordered_bias

    return linear_module



@torch.no_grad()
def get_sparse_attn_mask_from_topk(x, topk, recent_tokens=256, sink_tokens=128):
    seq_len = x.size(-1)
    topk = min(topk, seq_len)
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    if sink_tokens > 0:
        mask[..., :, :sink_tokens] = True
    if recent_tokens > 0:
        mask[..., :, -recent_tokens:] = True
    x_clone = x.clone()
    x_clone.masked_fill_(mask, torch.finfo(x.dtype).min)

    if topk - recent_tokens - sink_tokens <= 0:
       return mask, torch.Tensor([0.0])
    topk = topk - recent_tokens - sink_tokens # update
    _, topk_indices = torch.topk(x_clone, k=topk, dim=-1, sorted=False)

    mask.scatter_(-1, topk_indices, True)
    sparsity = 1 - mask.sum().float()/mask.numel()
    return mask, sparsity

@torch.no_grad()
def get_sparse_attn_mask_from_threshold(x, threshold, scale = 128**-0.5):
    prob = torch.softmax(x*scale, dim=-1)
    mask = prob > threshold
    sparsity = 1 - mask.sum().float() / mask.numel()
    return mask, sparsity

@torch.no_grad()
def get_sparse_attn_mask_from_topp(x, topp, scale = 128**-0.5):
    probs = torch.softmax(x * scale, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask_sorted = probs_sum - probs_sort <= topp
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(dim=-1, index=probs_idx, src=mask_sorted)
    sparsity = 1 - mask.sum().float() / mask.numel()
    return mask, sparsity

@torch.no_grad()
def get_sparse_attn_mask_from_radio(x, radio):
    seq_len = x.size(-1)
    topk = int((1-radio) * seq_len)
    return get_sparse_attn_mask_from_topk(x, topk)

@torch.no_grad()
def get_sparse_block_indices_from_topk(x, topk):
    seq_len = x.size(-1)
    if seq_len < topk:
        topk = seq_len
    _, topk_indices = torch.topk(x, k=topk, dim=-1, sorted=False)
    return topk_indices.to(dtype=torch.int32)


def reorder_model_linear_weights(model, full_attention_heads):
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = full_attention_heads[idx]
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )



if __name__ == "__main__":
    a = torch.tensor([[0.10, 0.40, 0.05, 0.25, 0.15, 0.05]])
    print(get_sparse_attn_mask_from_topp(a,0.75))