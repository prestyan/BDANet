# 导入必要的库
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


# 定义超参数

# 定义通道注意力模块
class ChannelWiseAttention(nn.Module):
    def __init__(self, C, weight_decay=0.00004):
        super(ChannelWiseAttention, self).__init__()
        self.C = C
        self.weight_decay = weight_decay

        # 定义参数
        self.weight = nn.Parameter(torch.empty(C, C))
        self.bias = nn.Parameter(torch.zeros(C))

        self.dfc = nn.Linear(C, int(C / 2))
        self.ufc = nn.Linear(int(C / 2), C)

        # 初始化参数
        nn.init.orthogonal_(self.weight)
        # 或者直接赋值
        # self.weight.data = torch.randn(C, C)

        # 定义正则化
        self.regularizer = nn.L1Loss()

    def forward(self, feature_map):
        # 获取批次大小和序列长度 [16, 61, 1, 500]
        N, C, H, W = feature_map.size()

        # 对特征图进行平均池化，得到通道向量
        channel_vector = F.avg_pool2d(feature_map, (H, W))
        channel_vector = channel_vector.view(N, C)
        channel_vector = F.log_softmax(self.ufc(F.tanh(self.dfc(channel_vector))), dim=1)
        channel_attention = channel_vector.view(N, C, 1, 1).repeat(1, 1, H, W)
        attended_fm = feature_map * channel_attention
        reg_loss = self.regularizer(self.weight, torch.zeros_like(self.weight)) * self.weight_decay
        '''
        # 对通道向量进行线性变换和sigmoid激活，得到通道注意力向量
        channel_attention = torch.sigmoid(torch.matmul(channel_vector.view(N, C), self.weight) + self.bias)

        # 对通道注意力向量进行复制和重塑，得到通道注意力矩阵
        channel_attention = channel_attention.view(N, C, 1, 1).repeat(1, 1, H, W)

        # 对特征图和通道注意力矩阵进行点乘，得到通道注意力特征图
        attended_fm = feature_map * channel_attention

        # 计算正则化损失
        reg_loss = self.regularizer(self.weight, torch.zeros_like(self.weight)) * self.weight_decay
        '''
        self.attention = channel_attention
        return attended_fm, channel_attention

    def get_attention(self):
        return self.attention


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1)
        y = self.avg_pool(x)
        y = self.se(y)
        # return x * y.expand_as(x)
        return x * y


# 定义一个多维注意力模型
class MultiDimensionalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 初始化模型参数
        super().__init__()
        self.W1 = nn.Linear(input_dim, output_dim, True)  # W1矩阵
        self.W2 = nn.Linear(input_dim, hidden_dim, True)  # W2矩阵
        self.W = nn.Linear(hidden_dim, output_dim, True)  # W矩阵

    def forward(self, h, q):
        # 前向传播计算
        z = self.W(F.elu(self.W1(h) + self.W2(q)))  # 根据公式计算z
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


#
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


# GELU激活函数
class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,  # 0.5调参
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class MDAOutput(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mda = MultiDimensionalAttention(input_dim, hidden_dim, output_dim)
        self.q_linear = nn.Linear(input_dim, input_dim)

    def forward(self, h):
        q = self.q_linear(h)
        z = self.mda(h, q)
        p = generate_p(z, h)
        return torch.matmul(p.repeat(1, 1, 32).transpose(1, 2), h)


def generate_p(z, h):
    sl = z.size(1)
    probability = torch.matmul(z.unsqueeze(2), h.unsqueeze(2).transpose(2, 3)).squeeze(2)
    total_probability = torch.sum(probability, dim=1)
    probability = torch.div(probability, total_probability.unsqueeze(1))
    return probability
