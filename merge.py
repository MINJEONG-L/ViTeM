
from typing import Tuple, Callable
import torch.nn.functional as F
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt
import os, json
import torch
import math
import torch
from typing import Optional


SIM_CACHE = {} 
import torch
def do_nothing(x: torch.Tensor, mode:str=None):
    return x

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def reset_cs_logs():
    for k in CS_LOGS:
        CS_LOGS[k].clear()

WINDOW_CS_LOGS = {
    "pre": [],   # list of 1D tensors (W,)
    "post": [],  # list of 1D tensors (W,)
    "delta": []  # list of 1D tensors (W,)
}

FLOP_LOG = {
    "sim_local": 0,   # local cosine (window self-sim)
    "sim_text": 0,    # cross similarity (Q_text x K_text)
    "cache_hit": 0,
    "cache_miss": 0,
}
def _add_flops(key, val):
    if key in FLOP_LOG:
        FLOP_LOG[key] += int(val)
    else:
        FLOP_LOG[key] = int(val)




_REP_LOGS = []

def _rep_flat_to_xy(idx_flat, W):
    y = (idx_flat // W).detach().cpu().tolist()
    x = (idx_flat %  W).detach().cpu().tolist()
    return y, x

CURRENT_PROMPT_ID = "(unknown)"
CURRENT_BLOCK_NAME = "(unknown)"
_REP_LOGS = []

def log_representatives(*, H, W, window_hw, xs, ys, scores=None):
    _REP_LOGS.append({
        "block": CURRENT_BLOCK_NAME,
        "prompt_id": CURRENT_PROMPT_ID,
        "H": int(H), "W": int(W),
        "window": [int(window_hw[0]), int(window_hw[1])],
        "x": [int(v) for v in xs],
        "y": [int(v) for v in ys],
        "scores": [float(s) for s in (scores or [])],
    })

def flush_rep_logs(path="rep_logs/rep_positions.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for row in _REP_LOGS:
            f.write(json.dumps(row) + "\n")
    _REP_LOGS.clear()

REP_LOG_PATH = "rep_logs/rep_positions.jsonl"
os.makedirs(os.path.dirname(REP_LOG_PATH), exist_ok=True)

def _flush_rep_log(rec):
    with open(REP_LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")

def reset_window_cs_logs():
    for k in WINDOW_CS_LOGS:
        WINDOW_CS_LOGS[k].clear()
        
def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

cached_cosine_sim = None  # 이전 timestep의 유사도 행렬 저장
t = 10

def generate_grouped_indices(h: int, w: int, sy: int, sx: int, device: torch.device = "cuda:0"):
    """
    전체 토큰을 sy x sx 윈도우로 나누는 인덱스 생성
    return shape: [1, num_windows, tokens_per_window]
    """
    flat_indices = torch.arange(h * w, device=device).reshape(h, w)
    patches = flat_indices.unfold(0, sy, sy).unfold(1, sx, sx)  # [H/sy, W/sx, sy, sx]
    grouped = patches.contiguous().view(-1, sy * sx)            # [num_windows, 4]
    grouped = grouped.unsqueeze(0)                              # [1, 1024, 4]
    return grouped



# 원하는 옵션 설정 (예: 줄 수, 너비 등)
torch.set_printoptions(threshold=1000000, linewidth=20000)


number = 0

def bipartite_soft_matching_max_cosine2d(metric: torch.Tensor,
                                         w: int, h: int, sx: int, sy: int, r: int,
                                         text_embedding: torch.Tensor = None,
                                         text_weight: float = 0.3,
                                         generator: torch.Generator = None,
                                         final_layer_norm: torch.nn.LayerNorm = None,
                                         q_proj: nn.Linear = None,
                                         k_proj: nn.Linear = None,
                                         block_name: str = "(unknown)",    
                                         prompt_id: str = "default",        
                                         step: int = -1    
                                         ) -> Tuple[Callable, Callable]:
    """
    text prompt와 로컬 유사도를 통합한 representative token 선택 방식
    """
    B, N, C = metric.shape
    device = metric.device
    global t
    global number
    gather = mps_gather_workaround if device.type == "mps" else torch.gather
    grouped_indices = generate_grouped_indices(h, w, sy, sx, device)  # [1, 1024, 4]
    grouped_indices = grouped_indices.expand(B, -1, -1)               # [B, 1024, 4]

    B, N, C = metric.shape                    # [4, 4096, 320]
    _, W, T = grouped_indices.shape          # [4, 1024, 4]

    global SIM_CACHE
    cache_key = None
    if block_name is not None and step is not None and step >= 0:
        cache_key = (str(block_name), int(step))

    if cache_key is not None and cache_key in SIM_CACHE:
        total_score = SIM_CACHE[cache_key]
        FLOP_LOG["cache_hit"] += 1
    else:
        flat_indices = grouped_indices.view(B, -1)  # [4, 4096]

        gathered = metric.gather(dim=1, index=flat_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, 4096, C]

        window_tokens = gathered.view(B, W, T, C)
        print("window size ", W)
        normed_tokens = window_tokens / window_tokens.norm(dim=-1, keepdim=True)
        local_sim = normed_tokens @ normed_tokens.transpose(-1, -2)  # [B, 1024, 4, 4]
        local_score = local_sim.mean(dim=-1)              
        B,W,T,C_block = window_tokens.shape   # window_tokens: [B,W,T,C]
        _add_flops("sim_local", 2 * B * W * T * T * C_block)           # [B, 1024, 4]

        if text_embedding is not None and q_proj is not None:
            B, W, T, C_block = window_tokens.shape 
            if text_embedding.shape[0] == 1 and B > 1:
                text_embedding = text_embedding.expand(B, -1, -1).contiguous()
            q = torch.matmul(window_tokens, q_proj.weight.t().to(window_tokens.dtype))
            if q_proj.bias is not None:
                # print("bias is not none")
                q = q + q_proj.bias.to(q.dtype)     # [B, W, T, Cq]

            if k_proj is not None:
                k = torch.matmul(text_embedding, k_proj.weight.t().to(text_embedding.dtype))
                if k_proj.bias is not None:
                    k = k + k_proj.bias.to(k.dtype)
            else:
                k = text_embedding 
            
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(k, dim=-1)

            sim = torch.einsum('bwtc,blc->bwtl', qn.to(kn.dtype), kn)
            # local_sim: [B,W,T,T]  (윈도우 self-sim)
# sim      : [B,W,T,L]  (cross sim)
            t += 1
            text_score = sim.max(dim=-1).values
            if step is not None and step >= 0:
                # w, h는 latent의 전체 크기. (e.g., 64, 64)
                save_stitched_cross_attention_map(
                    sim_tensor=sim, 
                    step=step,
                    w_total_pixels=w,
                    h_total_pixels=h,
                    save_dir="cross_sim_heatmaps_over_time"
                )
        else:
            text_score = torch.zeros_like(local_score)
        L_text = text_embedding.shape[1]
        C_q = q.shape[-1]   # after q_proj
        _add_flops("sim_text", 2 * B * W * T * L_text * C_q)
        
        total_score = (1 - text_weight) * local_score + text_weight * text_score

    if cache_key is not None:
        SIM_CACHE[cache_key] = total_score
        FLOP_LOG["cache_miss"] += 1
    max_sim_idx = total_score.argmin(dim=-1)  # [B, 1024]

    dst_global_idx = grouped_indices.gather(2, max_sim_idx.unsqueeze(-1)).squeeze(-1)  # [B, 1024]


    idx_buffer = torch.zeros(B, N, device=device, dtype=torch.int64)
    for b in range(B):
        idx_buffer[b, dst_global_idx[b]] = -1

    rand_idx = idx_buffer.argsort(dim=1)  # [B, N]
    # num_dst = hsy * wsx
    num_dst = W
    a_idx = rand_idx[:, num_dst:]
    b_idx = rand_idx[:, :num_dst]

    def split(x):
        C = x.shape[-1]
        src = torch.stack([x[b].index_select(0, a_idx[b]) for b in range(B)])  # [B, N-r, C]
        dst = torch.stack([x[b].index_select(0, b_idx[b]) for b in range(B)])  # [B, r, C]
        return src, dst

    a, b = split(F.normalize(metric, dim=-1))  # [B, N-r, C], [B, r, C]
    scores = torch.matmul(a, b.transpose(1, 2))  # [B, N-r, r]
    

    r = min(a.shape[1], r)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    unm_idx = edge_idx[..., r:, :]
    src_idx = edge_idx[..., :r, :]
    dst_idx = torch.gather(node_idx.unsqueeze(-1), dim=1, index=src_idx)
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        dst_weight = 0.9
        unm = torch.gather(src, dim=1, index=unm_idx.expand(n, t1 - r, c))
        src = torch.gather(src, dim=1, index=src_idx.expand(n, r, c))
        dst_original = dst.clone()
        dst_with_src = dst.scatter(1, dst_idx.expand(n, r, c), src)
        dst_weighted = dst_original * dst_weight + dst_with_src * (1 - dst_weight)

        return torch.cat([unm, dst_weighted], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape
        src = torch.gather(dst, dim=1, index=dst_idx.expand(B, r, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(1, b_idx.unsqueeze(-1).expand(B, num_dst, c), dst)
        out.scatter_(1, a_idx.gather(1, unm_idx.squeeze(-1)).unsqueeze(-1).expand(B, unm_len, c), unm)
        out.scatter_(1, a_idx.gather(1, src_idx.squeeze(-1)).unsqueeze(-1).expand(B, r, c), src)
        return out

    return merge, unmerge


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        del idx_buffer, idx_buffer_view
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        r = min(a.shape[1], r)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge
