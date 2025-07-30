import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any


# 总模块--Kronecker-Hyper-Transport-Refiner(KronHTR)
# ---------------------------------------------------------------------------
# 模块一: 格罗莫夫-瓦瑟斯坦关系对齐 (GW-RA)
# ---------------------------------------------------------------------------

def compute_gromov_wasserstein(
        f_seq: torch.Tensor,
        f_struct: torch.Tensor,
        epsilon: float = 1e-3,
        max_iter: int = 50,
        gw_iter: int = 5
) -> torch.Tensor:
    """
    计算序列特征和结构特征之间的格罗莫夫-瓦瑟斯坦(GW)最优传输计划。
    该函数实现了带熵正则化的GW距离计算，通过迭代的Sinkhorn算法求解。

    Args:
        f_seq (torch.Tensor): 主分支的序列特征 (B, L, D)。
        f_struct (torch.Tensor): 从分支的结构特征 (B, L, D)。
        epsilon (float): Sinkhorn算法中的熵正则化项。
        max_iter (int): Sinkhorn算法内部循环的最大迭代次数。
        gw_iter (int): GW算法外部循环的最大迭代次数。

    Returns:
        torch.Tensor: 传输计划矩阵 Gamma (B, L, L)。
    """
    B, L, _ = f_seq.shape
    device = f_seq.device

    # 1. 计算模态内部的距离矩阵 (使用余弦距离)
    # 归一化特征以计算余弦相似度
    f_seq_norm = F.normalize(f_seq, p=2, dim=-1)
    f_struct_norm = F.normalize(f_struct, p=2, dim=-1)

    # (B, L, D) @ (B, D, L) -> (B, L, L)
    cos_sim_seq = torch.bmm(f_seq_norm, f_seq_norm.transpose(1, 2))
    cos_sim_struct = torch.bmm(f_struct_norm, f_struct_norm.transpose(1, 2))

    # 距离 = 1 - 相似度
    dist_seq = 1 - cos_sim_seq
    dist_struct = 1 - cos_sim_struct

    # 2. 初始化概率分布和传输计划
    p = torch.full((B, L), 1.0 / L, device=device)
    q = torch.full((B, L), 1.0 / L, device=device)
    gamma = torch.einsum('bi,bj->bij', p, q)  # 初始为独立联合分布

    # 3. 迭代求解GW
    # 预计算 L(C_s, C_t) = C_s^2 * p * 1^T + 1 * q^T * C_t^2
    # 这是针对L2损失的优化计算
    c1_pow2 = dist_seq.pow(2)
    c2_pow2 = dist_struct.pow(2)
    const_mtx = torch.einsum('bij,bk->bik', c1_pow2, q) + torch.einsum('bi,bjk->bjk', p, c2_pow2)

    for _ in range(gw_iter):
        # 计算伪代价矩阵
        cost_matrix = const_mtx - 2 * torch.einsum('bik,bkl,bjl->bij', dist_seq, gamma, dist_struct)

        # 4. 使用Sinkhorn算法求解内部的OT问题
        kernel = torch.exp(-cost_matrix / epsilon)
        u = torch.ones_like(p)

        for _ in range(max_iter):
            v = q / torch.einsum('bij,bi->bj', kernel, u)
            u = p / torch.einsum('bij,bj->bi', kernel, v)

        gamma = torch.einsum('bi,bij,bj->bij', u, kernel, v)

    return gamma


# ---------------------------------------------------------------------------
# 模块二: 基于克罗内克积的超网络条件生成 (Kron-HSC)
# ---------------------------------------------------------------------------

class KronHypernetwork(nn.Module):
    """
    克罗内克超网络。
    以结构特征为输入，为每个精炼步骤动态生成克罗内克积因子。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dim = config['dim']
        self.num_refinement_steps = config['num_refinement_steps']
        self.kron_dims = config['kron_dims']  # (da, db)
        da, db = self.kron_dims

        assert da * db == self.dim, "Kronecker dimensions must multiply to feature dimension"

        # 聚合网络，将 (B, L, D) 的结构特征聚合成 (B, D) 的条件向量
        self.aggregator = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
        )

        # 为每个精炼步骤生成克罗内克因子的头部网络
        # 输出维度: num_steps * (da*da + db*db)
        total_kron_params = self.num_refinement_steps * (da * da + db * db)
        self.kron_generator = nn.Sequential(
            nn.Linear(self.dim, 512),
            nn.GELU(),
            nn.Linear(512, total_kron_params)
        )

    def forward(self, f_struct: torch.Tensor) -> list:
        """
        Args:
            f_struct (torch.Tensor): 从分支的结构特征 (B, L, D)。

        Returns:
            list]: 包含K个元组的列表，
            每个元组是 (K_A_k, K_B_k)，形状分别为 (B, da, da) 和 (B, db, db)。
        """
        B = f_struct.shape
        da, db = self.kron_dims

        # 1. 聚合结构特征
        struct_pooled = torch.mean(f_struct, dim=1)  # (B, D)
        condition_vector = self.aggregator(struct_pooled)  # (B, D)

        # 2. 生成所有克罗内克因子
        kron_params = self.kron_generator(condition_vector)  # (B, K * (da^2 + db^2))

        # 3. 重塑参数为每个步骤的因子矩阵
        kron_factors =
        current_pos = 0
        for _ in range(self.num_refinement_steps):
            # 提取 K_A 的参数并重塑
            k_a_params = kron_params[:, current_pos: current_pos + da * da]
            k_a = k_a_params.view(B, da, da)
            current_pos += da * da

            # 提取 K_B 的参数并重塑
            k_b_params = kron_params[:, current_pos: current_pos + db * db]
            k_b = k_b_params.view(B, db, db)
            current_pos += db * db

            kron_factors.append((k_a, k_b))

        return kron_factors


# ---------------------------------------------------------------------------
# 模块三: 克罗内克积迭代精炼 (Kron-IR)
# ---------------------------------------------------------------------------

def efficient_kron_update(x: torch.Tensor, k_a: torch.Tensor, k_b: torch.Tensor) -> torch.Tensor:
    """
    使用克罗内克积的性质高效地计算更新量 (K_A ⊗ K_B)x。
    避免了显式构造 D x D 的巨大矩阵。
    公式: vec(K_B @ mat(x) @ K_A.T)

    Args:
        x (torch.Tensor): 输入特征 (B, L, D)。
        k_a (torch.Tensor): 克罗内克因子A (B, da, da)。
        k_b (torch.Tensor): 克罗内克因子B (B, db, db)。

    Returns:
        torch.Tensor: 更新后的特征 (B, L, D)。
    """
    B, L, D = x.shape
    da, db = k_a.shape[1], k_b.shape[1]

    # 1. 将输入特征向量重塑为矩阵 mat(x)
    # (B, L, D) -> (B, L, db, da)
    x_reshaped = x.view(B, L, db, da)

    # 2. 计算 K_B @ mat(x)
    # (B, db, db) @ (B, L, db, da) -> (B, L, db, da)
    # 使用einsum实现批处理矩阵乘法
    step1 = torch.einsum('bij,bljk->blik', k_b, x_reshaped)

    # 3. 计算 (K_B @ mat(x)) @ K_A.T
    # (B, L, db, da) @ (B, da, da) -> (B, L, db, da)
    # K_A.T 的 einsum 表达式为 'bji'
    step2 = torch.einsum('blik,bji->bljk', step1, k_a)

    # 4. 将结果展平回向量 vec(...)
    # (B, L, db, da) -> (B, L, D)
    update = step2.contiguous().view(B, L, D)

    return update


class KronRefinementBlock(nn.Module):
    """
    单个克罗内克积精炼块。
    包含一个受GW计划矩阵调制的轻量级自注意力和克罗内克更新。
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))  # 可学习的调制强度

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, kron_factors: Tuple) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入序列特征 (B, L, D)。
            gamma (torch.Tensor): GW传输计划矩阵 (B, L, L)。
            kron_factors (Tuple): (K_A, K_B) 克罗内克因子。

        Returns:
            torch.Tensor: 精炼后的序列特征 (B, L, D)。
        """
        B, L, D = x.shape

        # --- 1. GW调制的自注意力 ---
        x_res = x
        x = self.norm1(x)

        q,k,v = self.qkv_proj(x).chunk(3, dim=-1)
        #q, k, v =

        # 计算注意力分数并加入GW调制
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 将gamma广播到每个注意力头
        gamma_broadcast = gamma.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores + F.logsigmoid(gamma_broadcast)  # 使用logsigmoid增加数值稳定性

        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        x = x_res + attn_output

        # --- 2. 克罗内克积更新 ---
        x_res = x
        x = self.norm2(x)

        k_a, k_b = kron_factors
        kron_update = efficient_kron_update(x, k_a, k_b)

        x = x_res + self.scale * kron_update

        return x


class KronIterativeRefiner(nn.Module):
    """
    执行K步迭代精炼的模块。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_refinement_steps = config['num_refinement_steps']
        self.refinement_blocks = nn.ModuleList(, num_heads = config.get('num_heads', 4))
        for _ in range(self.num_refinement_steps)
    ])

    def forward(self, f_seq: torch.Tensor, kron_factors: list, gamma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_seq (torch.Tensor): 初始序列特征 (B, L, D)。
            kron_factors (list): K个克罗内克因子元组的列表。
            gamma (torch.Tensor): GW传输计划矩阵 (B, L, L)。

        Returns:
            torch.Tensor: 最终精炼后的序列特征 (B, L, D)。
        """
        x = f_seq
        for i in range(self.num_refinement_steps):
            x = self.refinement_blocks[i](x, gamma, kron_factors[i])
        return x


# ---------------------------------------------------------------------------
# 主模型: Kron-HTR
# ---------------------------------------------------------------------------

class KronHTR(nn.Module):
    """
    完整的 Kron-HTR 模型----Kronecker-Hyper-Transport-Refiner。
    集成了GW对齐、克罗内克超网络和迭代精炼三个模块。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # 模块二: 克罗内克超网络
        self.kron_hypernetwork = KronHypernetwork(config)

        # 模块三: 迭代精炼器
        self.iterative_refiner = KronIterativeRefiner(config)

    def forward(self, f_seq: torch.Tensor, f_struct: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。

        Args:
            f_seq (torch.Tensor): 主分支(ESM-Mamba)的序列特征 (B, L, D)。
            f_struct (torch.Tensor): 从分支(Saprot-GNN)的结构特征 (B, L, D)。

        Returns:
            torch.Tensor: 最后的融合输出f_refined。
        """
        # 模块一: 计算GW传输计划 (在forward中动态计算)
        # 注意：在实际训练中，如果GW计算成为瓶颈，可以考虑缓存或降低计算频率
        with torch.no_grad():  # 通常GW计算不参与反向传播
            gamma = compute_gromov_wasserstein(f_seq.detach(), f_struct.detach())

        # 模块二: 超网络生成克罗内克因子
        kron_factors = self.kron_hypernetwork(f_struct)

        # 模块三: 迭代精炼序列特征
        f_refined = self.iterative_refiner(f_seq, kron_factors, gamma)

        return f_refined
