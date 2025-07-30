import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


# ---------------------------------------------------------------------------- #
#                             核心创新模块：DSFN                                 #
# ---------------------------------------------------------------------------- #

class LoRALayer(nn.Module):
    """
    低秩适应层 (Low-Rank Adaptation Layer)。
    该层用于以极高的参数效率来近似一个全尺寸的线性层。
    它包含一个冻结的基底权重和一个可训练的低秩更新矩阵 (B*A)。
    参考: "LoRA: Low-Rank Adaptation of Large Language Models" [5, 7]。
    """

    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        self.r = rank
        self.alpha = alpha

        # 基底权重，在实际应用中可以加载预训练权重，此处为简化设为不可训练
        self.base_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.base_layer.weight.requires_grad = False

        # 低秩分解矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # 基底权重的前向传播 + 低秩适应矩阵的前向传播
        base_output = self.base_layer(x)
        lora_update = (x @ self.lora_A @ self.lora_B) * self.alpha
        return base_output + lora_update


class StructurallyGuidedDeformConv1d(nn.Module):
    """
    结构引导的可变形1D卷积 (Structurally-Guided Deformable Convolution, SGDC)。
    这是一个创新的1D可变形卷积实现，其采样偏移量 (offsets) 由外部的辅助特征提供，
    而非从主特征自身学习。这实现了辅助信息对主特征提取过程的直接引导。
    参考: "Deformable Convolutional Networks" [2, 3]。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(StructurallyGuidedDeformConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 标准的卷积权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset):
        """
        前向传播。
        :param x: 主特征输入, 形状
        :param offset: 由辅助特征生成的偏移量, 形状
        :return: 卷积后的特征, 形状
        """
        B, _, L_in = x.shape
        L_out = offset.shape[1]

        # 1. 生成采样网格
        # 创建一个规则的采样网格，代表卷积核的每个位置
        kernel_range = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=x.device, dtype=x.dtype)
        regular_grid = kernel_range.view(1, -1, 1) * self.dilation  # [1, K, 1]

        # 创建输出位置的网格
        output_grid_range = torch.arange(L_out, device=x.device, dtype=x.dtype)
        output_grid = output_grid_range.view(1, 1, -1) * self.stride + self.padding  # [1, 1, L_out]

        # 2. 应用偏移量
        # 将规则网格与输出位置网格相加，然后加上外部提供的偏移量
        # 得到最终的、不规则的采样坐标
        sampling_grid = regular_grid + output_grid + offset  # 广播机制:

        # 3. 线性插值采样
        # 将采样坐标归一化到 [-1, 1] 范围
        sampling_grid_norm = 2 * sampling_grid / (L_in - 1) - 1

        # 添加一个维度以匹配 grid_sample 的期望输入格式
        # grid_sample 需要 (N, H, W, 2) for 2D, 我们模拟为 (N, 1, L_out, 2)
        # 其中一个维度是我们的采样点，另一个维度是固定的0
        sampling_grid_norm = sampling_grid_norm.permute(0, 2, 1).unsqueeze(1)  #
        grid_zeros = torch.zeros_like(sampling_grid_norm)
        final_grid = torch.stack([sampling_grid_norm, grid_zeros], dim=-1)  #

        # x 需要被调整为 (N, C, 1, L_in) 以进行采样
        x_reshaped = x.unsqueeze(2)  #

        # 使用 F.grid_sample 进行采样
        # 我们需要对每个卷积核位置进行采样
        sampled_features = []
        for i in range(self.kernel_size):
            # grid_sample 的 grid 参数形状为
            # 我们的 H_out=1, W_out=L_out
            grid_slice = final_grid[:, :, :, i, :].squeeze(-2)  #
            sampled_x = F.grid_sample(x_reshaped, grid_slice, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_features.append(sampled_x.squeeze(2))  #

        # 堆叠采样后的特征
        sampled_x_stack = torch.stack(sampled_features, dim=-1)  #
        sampled_x_stack = sampled_x_stack.permute(0, 1, 3, 2)  #

        # 4. 执行卷积
        # 将采样后的特征与卷积核进行计算
        # 调整形状以进行矩阵乘法
        if self.groups == 1:
            output = F.conv1d(
                sampled_x_stack.reshape(B, self.in_channels * self.kernel_size, L_out),
                self.weight.reshape(self.out_channels, self.in_channels * self.kernel_size, 1),
                bias=self.bias,
                stride=1,
                padding=0,
                groups=1
            )
        else:  # 分组卷积
            # 更复杂的实现，此处为简化，主要展示核心逻辑
            # 此处我们使用一个更直接的 einsum 实现分组卷积
            # * [G, C_out/G, C_in/G, K] ->
            x_grouped = rearrange(sampled_x_stack, 'b (g c_in_g) k l -> b g c_in_g k l', g=self.groups)
            w_grouped = rearrange(self.weight, '(g c_out_g) c_in_g k -> g c_out_g c_in_g k', g=self.groups)
            output = torch.einsum('bgikl,goik->bgol', x_grouped, w_grouped)
            output = rearrange(output, 'b g o l -> b (g o) l')
            if self.bias is not None:
                output += self.bias.view(1, -1, 1)

        return output


class LoRAModulatedInjectionUnit(nn.Module):
    """
    LoRA调制注入单元 (LoRA-Modulated Injection Unit, L-MIU)。
    使用LoRA层来高效地处理辅助特征，并解耦地生成两组控制信号：
    1. 可变形卷积的偏移量 (offsets)。
    2. FiLM层的调制参数 (gamma 和 beta)。
    """

    def __init__(self, d_model, kernel_size, lora_rank=4, lora_alpha=1.0):
        super().__init__()

        # 使用一个简单的CNN来初步处理辅助特征
        self.pre_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding='same')
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

        # 使用LoRA层生成偏移量，参数效率高
        # 输出维度是 kernel_size，因为每个卷积核位置都需要一个偏移量
        self.offset_generator = LoRALayer(d_model, kernel_size, rank=lora_rank, alpha=lora_alpha)

        # 使用LoRA层生成FiLM参数 (gamma 和 beta)
        # 输出维度是 d_model * 2，一半给gamma，一半给beta
        self.film_param_generator = LoRALayer(d_model, d_model * 2, rank=lora_rank, alpha=lora_alpha)

    def forward(self, aux_feature):
        """
        前向传播。
        aux_feature 的形状: [batch, seq_len, d_model]
        """
        # 调整维度以适应Conv1d: ->
        x = aux_feature.permute(0, 2, 1)
        x = self.pre_conv(x)

        # 调整回 以进行后续处理
        x = x.permute(0, 2, 1)
        x = self.norm(self.act(x))

        # 1. 生成可变形卷积的偏移量
        # 输出形状: ->
        offsets = self.offset_generator(x).permute(0, 2, 1)

        # 2. 生成FiLM参数
        film_params = self.film_param_generator(x)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        return offsets, gamma, beta


class DSFN(nn.Module):
    """
    DSF-Net：具有结构引导的解耦调制融合网络。Dominant-Stream Fusion Network(DSFN)
    该网络集成了三大创新：
    1. 结构引导的可变形卷积 (SGDC)：辅助特征指导主特征的采样过程。
    2. LoRA调制注入单元 (L-MIU)：以极高参数效率生成控制信号。
    3. 双阶解耦FiLM融合：在采样和特征层面进行两阶段的深度融合。
    """

    def __init__(self, d_model=1280, kernel_size=5, lora_rank=8, lora_alpha=1.0):
        super().__init__()

        # 主特征处理流：一个由辅助特征引导的可变形卷积层
        # padding = (kernel_size - 1) // 2 确保输出长度不变
        self.sgdc_block = StructurallyGuidedDeformConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.main_norm = nn.LayerNorm(d_model)

        # 辅助特征处理流：生成偏移量和FiLM参数
        self.l_miu = LoRAModulatedInjectionUnit(
            d_model=d_model,
            kernel_size=kernel_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )

    def forward(self, main_feature, auxiliary_feature):
        """
        前向传播。
        :param main_feature: 主特征张量，形状。
        :param auxiliary_feature: 辅助特征张量，形状。
        :return: 融合后的特征张量，形状。
        """
        # 1. 辅助流生成两组解耦的控制信号
        offsets, gamma, beta = self.l_miu(auxiliary_feature)

        # 2. 主特征流进行第一阶段融合：采样引导
        # 调整维度以适应Conv1d: ->
        main_feature_permuted = main_feature.permute(0, 2, 1)

        # SGDC模块接收主特征和来自辅助特征的偏移量
        refined_main_permuted = self.sgdc_block(main_feature_permuted, offsets)

        # 调整回
        refined_main = refined_main_permuted.permute(0, 2, 1)
        refined_main = self.main_norm(refined_main)

        # 3. 第二阶段融合：特征调制 (FiLM)
        # FiLM: Fused = gamma * Refined_Main + beta
        # 参考: "FiLM: Visual Reasoning with a General Conditioning Layer" [10, 12]
        fused_feature = refined_main * gamma + beta

        return fused_feature
