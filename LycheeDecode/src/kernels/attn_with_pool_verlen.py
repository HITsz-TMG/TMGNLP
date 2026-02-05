import argparse
import time
import tilelang
import tilelang.language as T
import math
import torch
import torch.nn.functional as F

tilelang.disable_cache()

@tilelang.jit(out_idx=[-1])
def attn_with_pooling(
        batch, heads_kv, dim, max_cache_seqlen, block_N, block_H=16, threads=128,
        dtype="float16", accum_dtype="float"
):
    num_blocks = math.ceil(max_cache_seqlen / block_N)

    shape_q = [batch, heads_kv, dim]
    shape_k = [batch, max_cache_seqlen, heads_kv, dim]
    shape_attn = [batch, heads_kv, num_blocks]

    @T.prim_func
    def attn_with_pooling_func(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            cache_seqlens: T.Tensor([batch], "int32"),
            pool_attn: T.Tensor(shape_attn, dtype)
    ):
        with T.Kernel(batch, heads_kv, num_blocks, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_H, dim), dtype)
            K_shared = T.alloc_shared((block_N, dim), dtype)
            attn_local = T.alloc_fragment((block_H, block_N), accum_dtype)
            pool_attn_score = T.alloc_fragment((block_H,), accum_dtype)

            bid = bx
            hid = by
            sid = bz

            pad_len = max_cache_seqlen - cache_seqlens[bid]

            if sid * block_N > pad_len - block_N:
                T.copy(Q[bid, hid:hid+block_H, :], Q_shared)

                T.copy(K[bid, sid * block_N: sid * block_N + block_N, hid, :], K_shared)

                T.clear(attn_local)

                T.gemm(
                    Q_shared,
                    K_shared,
                    attn_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow
                )

                for i,j in T.Parallel(block_H, block_N):
                    attn_local[i,j] = T.if_then_else(sid * block_N + j < pad_len, -T.infinity(accum_dtype), attn_local[i,j])
                T.reduce_max(attn_local, pool_attn_score, dim=-1)

                pool_attn[bid, hid, sid] = pool_attn_score[0]


            else:
                pool_attn[bid, hid, sid] = -T.infinity(accum_dtype)

    return attn_with_pooling_func


def ref_program_torch(q, K, cache_seqlens, block_size):
    q_unsqueeze = q.unsqueeze(2)
    K_permuted = K.permute(0, 2, 1, 3)
    attn_scores = torch.matmul(q_unsqueeze, K_permuted.transpose(-2, -1))
    attn_scores_squeeze = attn_scores.squeeze(2)

    batch_size, max_cache_seqlen, _, _ = K.shape
    pad_lens = max_cache_seqlen - cache_seqlens
    indices = torch.arange(max_cache_seqlen, device=q.device).expand(batch_size, -1)
    mask = indices >= pad_lens.unsqueeze(1)
    attn_scores_squeeze.masked_fill_(~mask.unsqueeze(1), -torch.inf)

    return F.max_pool1d(
        attn_scores_squeeze,
        kernel_size=block_size,
        stride=block_size,
        ceil_mode=True
    )



def main(args):
    batch = args.batch
    heads_kv = args.heads_kv
    dim = args.dim
    max_cache_seqlen = args.max_cache_seqlen
    block_size = args.block_size

    dtype = torch.float16
    q = torch.randn(batch, heads_kv, dim, dtype=dtype).cuda()
    K = torch.randn(batch, max_cache_seqlen, heads_kv, dim, dtype=dtype).cuda()
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32, device='cuda')
    print(cache_seqlens)

    ref_output = ref_program_torch(q, K, cache_seqlens, block_size)
    print("ref output:", ref_output)
    print("ref output shape:", ref_output.shape)

    kernel = attn_with_pooling(batch, heads_kv, dim, max_cache_seqlen, block_N=block_size)

    kernel_output = kernel(q, K, cache_seqlens)
    print("kernel output:", kernel_output)
    print("kernel output shape:", kernel_output.shape)

    # max_diff = torch.max(torch.abs(kernel_output - ref_output))
    # print("max diff:", max_diff)


    torch.testing.assert_close(ref_output, kernel_output, rtol=1e-3, atol=1e-3)
    print("All check passed.")

    for _ in range(10):
        ref = ref_program_torch(q, K, cache_seqlens, block_size)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = ref_program_torch(q, K, cache_seqlens, block_size)
    torch.cuda.synchronize()
    print("ref time: ", (time.time() - start) / 100 * 1000)


    for _ in range(10):
        ref = kernel(q, K, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        ref = kernel(q, K, cache_seqlens)
    torch.cuda.synchronize()
    print("kernel time: ", (time.time() - start) / 100 * 1000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--heads_kv', type=int, default=8, help='heads_kv')
    parser.add_argument(
        '--max_cache_seqlen', type=int, default=32000, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--block_size', type=int, default=64, help='block_size')
    torch.manual_seed(42)
    args = parser.parse_args()
    main(args)