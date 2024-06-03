import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj, gaussian_Adj


def clones(module, N):  # 它接受两个参数：`module` 是要复制的神经网络层，`N` 是要生成的相同层的数量
    """
    Produce N identical layers.    # 生成N个相同的层
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList   # 表示生成的相同层的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    # `clones` 函数用于生成指定数量的相同的神经网络层，并将它们存储在一个 `nn.ModuleList` 对象中返回


def subsequent_mask(size):  # 它接受一个参数 size，表示生成的遮蔽矩阵的大小
    """
    mask out subsequent positions. # 函数的目标：用于遮蔽未来位置
    :param size: int
    :return: (1, size, size)   维度为 (1, size, size) 的矩阵
    """
    attn_shape = (1, size, size)  # (批次大小,行数,列数)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # 生成了一个遮蔽矩阵
    return torch.from_numpy(subsequent_mask) == 0
    # 一个布尔型张量，1 表示可到达的位置，0 表示遮蔽的位置，即未来的

class feature_attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, rate=4):
        super(feature_attention, self).__init__()
        self.nconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.channel_attention = nn.Sequential(
            nn.Linear(out_channels, int(out_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(out_channels / rate), out_channels)
        )
        self.time_attention = nn.Sequential(
            nn.Linear(out_channels, int(out_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(out_channels / rate), out_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, int(out_channels / rate), kernel_size=(1, kernel_size),
                      padding=(0, (kernel_size - 1) // 2)),
            nn.BatchNorm2d(int(out_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels / rate), out_channels, kernel_size=(1, kernel_size),
                      padding=(0, (kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # x : B T N C
        x = x.permute(0, 3, 2, 1)  # [B, C, N, T]
        x = self.nconv(x)  # 扩展数据的特征维度
        b, c, n, t = x.shape
        # 通道增强
        x_permute = x.permute(0, 2, 3, 1)  # [B, N, T, C]
        x_att_permute = self.channel_attention(x_permute)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  # [B, C, N, T]
        x = x * x_channel_att

        # 时间增强
        time_att = self.time_attention(x)  # 使用调整后的 x 作为输入
        x = x * time_att

        # 空间增强
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out.permute(0, 3, 2, 1)  # B T N C

class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # 对称归一化的邻接矩阵 (N, N)
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)  # 线性变换参数 Theta

    def forward(self, x):
        '''
        空间图卷积操作
        :param x: (batch_size, N, F_in) 输入数据
        :return: (batch_size, N, F_out) 输出数据
        '''
        # 执行空间图卷积操作：乘以邻接矩阵并应用线性变换 Theta
        # 结果经过 ReLU 激活函数
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # 对称归一化的邻接矩阵 (N, N)
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)  # 线性变换参数 Theta

    def forward(self, x):
        """
        空间图卷积操作
        :param x: (batch_size, N, T, F_in) 输入数据
        :return: (batch_size, N, T, F_out) 输出数据
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        # 将输入维度重新排列，变成 (batch_size*num_of_timesteps, num_of_vertices, in_channels)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))

        # 执行空间图卷积操作：乘以邻接矩阵并应用线性变换 Theta
        # 结果经过 ReLU 激活函数
        # 最后重新排列维度，变成 (batch_size, num_of_timesteps, num_of_vertices, out_channels)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class Spatial_Attention_layer(nn.Module):
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        compute spatial attention scores   计算空间注意力分数
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        # 重新排列维度，将时间步和顶点维度交换，并将数据重新变为形状 (batch_size * num_of_timesteps, num_of_vertices, in_channels)
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        # 计算空间注意力分数，通过矩阵乘法得到注意力分数，使用math.sqrt(in_channels)来进行缩放
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))  # (b,t, N, N)


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        """
        Spatial graph convolution operations with zoomed self-attention mechanism
        带有缩放的自注意力机制的空间图卷积操作
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        """
        空间位置编码
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        """
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index  # 用于指定哪些时间步需要添加位置编码
        self.max_len = max_len  # 用于指定哪些时间步需要添加位置编码
        # computing the positional encodings once in log space  在日志空间中计算一次位置编码
        # 计算位置编码，使用正弦和余弦函数
        pe = torch.zeros(max_len, d_model)  # 创建一个全零的位置编码矩阵
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # 将位置编码的维度调整为(1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.  向模块添加持久缓冲区
        # This is typically used to register a buffer that should not to be considered a model parameter.
        # 这通常用于注册不应被视为模型参数的缓冲区

    def forward(self, x):
        """
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        """
        if self.lookup_index is not None:
            # 对应时间步的输入添加位置编码
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            # 对所有时间步的输入添加位置编码
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())
        # .detach() 方法可以将一个张量从计算图中分离出来使其成为一个新的张量，不会保留 dropout 操作的梯度信息


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    实现残差连接和层归一化的模块
    """

    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        """
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        """
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))


class PositionWiseGCNFeedForward(nn.Module):  # 位置明智GCN前馈
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x:  (B, N_nodes, T, F_in)
        :return: (B, N, T, F_out)
        """
        return self.dropout(F.relu(self.gcn(x)))


def attention(query, key, value, mask=None, dropout=None):
    """

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    """
    d_k = query.size(-1)  # 获取键或查询向量的维度，通常用于缩放注意力权重
    # 计算注意力得分
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # 用 -1e9 填充 mask 为 0 的位置，将注意力得分设为负无穷
        # 这样在计算 softmax 时对应位置的注意力权重就会趋近于 0
    p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重

    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head  # 每个注意力头的维度
        self.h = nb_head  # 注意力头的数量
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 由 4 个线性层组成的一个列表
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        实现多头自注意力机制
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), 相同的遮蔽矩阵应用于所有 H 头.

        nbatches = query.size(0)  # 获取输入张量 query 的批次大小

        N = query.size(1)

        # (batch, N, T, d_model) --linear--> (batch, N, T, d_model)
        # --view--> (batch, N, T, h, d_k) --permute(2,3)--> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]  # zip 函数相当于按元素逐个匹配

        # apply attention on all the projected vectors in batch对批量中的所有投影向量应用注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)  # (batch, N, T1, d_model)


class Local_Global_Attention_en(nn.Module):  # 用于编码器
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3,
                 global_heads_ratio=1, dropout=.0):
        super(Local_Global_Attention_en, self).__init__()
        self.nb_head = nb_head
        self.local_heads_ratio = 1 - global_heads_ratio
        self.global_heads = int(nb_head * global_heads_ratio)
        self.local_heads = nb_head - self.global_heads
        self.head_dim = d_model // nb_head  # 每一个头的维度
        assert d_model % (self.global_heads + self.local_heads) == 0

        self.t_q_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)
        self.t_k_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)
        self.t_v_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)

        self.linear1 = (nn.Linear(d_model, int(d_model * self.local_heads_ratio)))  # for W^V
        self.linear2 = (nn.Linear(d_model, d_model))  # for W^O
        self.padding = (kernel_size - 1) // 2
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, int(d_model * self.local_heads_ratio), (1, kernel_size), padding=(0, self.padding)), 2)
        # 2 causal conv: 1  for query, 1 for key

        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, query_multi_segment=False, key_multi_segment=False):
        """
        Parameters
        ----------
        query : x
        key : x
        value : x
        mask
        query_multi_segment
        key_multi_segment
        Returns
        -------
        """
        nbatches = query.size(0)
        N = query.size(1)

        # 全局时间注意力
        t_q = self.t_q_conv(query.permute(0, 3, 1, 2))
        t_k = self.t_k_conv(key.permute(0, 3, 1, 2))
        t_v = self.t_v_conv(value.permute(0, 3, 1, 2))
        t_q = t_q.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_k = t_k.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_v = t_v.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_attn = (t_q @ t_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.dropout(t_attn)
        gt_x = (t_attn @ t_v).transpose(2, 3).contiguous().view(nbatches, N, -1, self.global_heads * self.head_dim)

        # 局部时间注意力
        # deal with key and query: temporal conv  处理键和查询：临时转换
        # (batch, N, T, d_model)-->permute(0, 3, 1, 2)-->(batch, d_model, N, T) --conv-->(batch, d_model, N, T)
        # --view-->(batch, h, d_k, N, T)--permute(0,3,1,4,2)-->(batch, N, h, T, d_k)
        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N,
                                                               -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context,
                                    (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N,
                                                               -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                        query[:, :, self.w_length:self.w_length + self.d_length, :],
                        key[:, :, self.w_length:self.w_length + self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N,
                                                               -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                        query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :],
                        key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)  # 在维度3上进行连接（时间维度上）
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [
                l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     4,
                                                                                                                     2)
                for
                l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and key_multi_segment:

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)
                                                           ).contiguous().view(nbatches, self.local_heads,
                                                                               self.head_dim,
                                                                               N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length,
                    :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1).permute(
                    0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.exit()

        # apply attention on all the projected vectors in batch

        # deal with value:
        # (batch,N,T,d_model)-linear->(batch, N, T, d_model)-view-> (batch, N, T, h, d_k)-permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linear1(value).view(nbatches, N, -1, self.local_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)  # scores: (batch, N, h, T1, T2)
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.dropout(p_attn)  # p_attn: (batch, N, h, T1, T2)
        lt_x = (p_attn @ value).transpose(2, 3).contiguous().view(nbatches, N, -1, self.local_heads * self.head_dim)
        # (batch, N, h, T1, d_k) -> (batch, N, T1, h, d_k) -> (batch, N, T1, d_model)

        x = torch.cat([gt_x, lt_x], dim=-1)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class Local_Global_Attention_de1(nn.Module):  # 用于解码器第二个注意力
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3,
                 global_heads_ratio=1, dropout=.0):
        super(Local_Global_Attention_de1, self).__init__()
        self.nb_head = nb_head
        self.local_heads_ratio = 1 - global_heads_ratio
        self.global_heads = int(nb_head * global_heads_ratio)
        self.local_heads = nb_head - self.global_heads
        self.head_dim = d_model // nb_head  # 每一个头的维度
        assert d_model % (self.global_heads + self.local_heads) == 0

        self.t_q_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)
        self.t_k_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)
        self.t_v_conv = nn.Conv2d(d_model, int(d_model * global_heads_ratio), kernel_size=1, bias=True)

        self.linear1 = (nn.Linear(d_model, int(d_model * self.local_heads_ratio)))  # for W^V
        self.linear2 = (nn.Linear(d_model, d_model))  # for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, int(d_model * self.local_heads_ratio),
                                                              (1, kernel_size),
                                                              padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, int(d_model * self.local_heads_ratio),
                                                            (1, kernel_size),
                                                            padding=(0, self.padding_1D))

        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, query_multi_segment=False, key_multi_segment=False):
        """
        Parameters
        ----------
        query : x
        key : x
        value : x
        query_multi_segment
        key_multi_segment
        Returns
        -------
        """
        nbatches = query.size(0)
        N = query.size(1)

        # 全局时间注意力
        t_q = self.t_q_conv(query.permute(0, 3, 1, 2))
        t_k = self.t_k_conv(key.permute(0, 3, 1, 2))
        t_v = self.t_v_conv(value.permute(0, 3, 1, 2))
        t_q = t_q.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_k = t_k.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_v = t_v.reshape(nbatches, self.global_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
        t_attn = (t_q @ t_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.dropout(t_attn)
        gt_x = (t_attn @ t_v).transpose(2, 3).contiguous().view(nbatches, N, -1, self.global_heads * self.head_dim)

        # 局部时间注意力
        # deal with key and query: temporal conv  处理键和查询：临时转换
        # (batch, N, T, d_model)-->permute(0, 3, 1, 2)-->(batch, d_model, N, T) --conv-->(batch, d_model, N, T)
        # --view-->(batch, h, d_k, N, T)--permute(0,3,1,4,2)-->(batch, N, h, T, d_k)
        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w = self.query_conv1Ds_aware_temporal_context(query[:, :, :self.w_length, :].permute(0, 3, 1, 2))[
                          :, :, :, :-self.causal_padding].contiguous().view(nbatches, self.local_heads, self.head_dim,
                                                                            N, -1).permute(
                    0, 3, 1, 4, 2)
                key_w = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads,
                                                                                        self.head_dim, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d = self.query_conv1Ds_aware_temporal_context(
                    query[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :,
                          :-self.causal_padding].contiguous().view(nbatches, self.local_heads, self.head_dim, N,
                                                                   -1).permute(0, 3, 1,
                                                                               4, 2)
                key_d = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(
                    query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(
                        0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.local_heads,
                                                                                       self.head_dim, N,
                                                                                       -1).permute(0, 3, 1,
                                                                                                   4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length,
                    :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1).permute(
                    0, 3, 1, 4, 2)

                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1).permute(
                0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(
                nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and key_multi_segment:

            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1
                                                             ).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.w_length > 0:
                key_w = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(
                        0, 3, 1, 2)).contiguous().view(nbatches, self.local_heads, self.head_dim, N, -1).permute(0, 3,
                                                                                                                 1, 4,
                                                                                                                 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.exit()

        # apply attention on all the projected vectors in batch

        # deal with value:
        # (batch,N,T,d_model)-linear->(batch, N, T, d_model)-view-> (batch, N, T, h, d_k)-permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linear1(value).view(nbatches, N, -1, self.local_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)  # scores: (batch, N, h, T1, T2)
        p_attn = F.softmax(scores, dim=-1)  # 计算注意力权重
        p_attn = self.dropout(p_attn)  # p_attn: (batch, N, h, T1, T2)
        lt_x = (p_attn @ value).transpose(2, 3).contiguous().view(nbatches, N, -1, self.local_heads * self.head_dim)
        # (batch, N, h, T1, d_k) -> (batch, N, T1, h, d_k) -> (batch, N, T1, d_model)

        x = torch.cat([gt_x, lt_x], dim=-1)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense  # 源数据的嵌入层
        # self.trg_embed = trg_dense  # 目标数据的嵌入层
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, src):
        """
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        """
        encoder_output = self.encode(src)  # (batch_size, N, T_in, d_model)

        return self.decode(encoder_output)

    def encode(self, src):
        """
        src: (batch_size, N, T_in, F_in)
        """
        h = self.src_embed(src)  # 将源数据进行嵌入
        return self.encoder(h)  # 编码
        # return self.encoder(self.src_embed(src))

    def decode(self, encoder_output):
        return self.prediction_generator(self.decoder(encoder_output))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gat, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        self.gat = gat
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)
        self.size = size

    def forward(self, x):
        """
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        """
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x,
                                 lambda x: self.self_attn(x, x, x, query_multi_segment=False, key_multi_segment=True))
            x = self.sublayer[1](x, self.gat)
            x = self.sublayer[2](x, self.feed_forward_gcn)
            return x
        else:
            x = self.self_attn(x, x, x, query_multi_segment=False, key_multi_segment=True)
            x = self.gat(x)
            x = self.feed_forward_gcn(x)
            return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer:  EncoderLayer   编码器层
        :param N:  int, number of EncoderLayers  编码器层数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        """
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, src_attn, gat, gcn, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        # self.self_attn = self_attn
        self.gat = gat
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 3)

    def forward(self, x):
        """
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        """
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[0](x, lambda x: self.src_attn(x, x, x, query_multi_segment=False,
                                                            key_multi_segment=True))  # output:  (batch, N, T', d_model)
            x = self.sublayer[1](x, self.gat)
            return self.sublayer[2](x, self.feed_forward_gcn)  # output: (batch, N, T', d_model)
        else:
            x = self.src_attn(x, x, x, query_multi_segment=False, key_multi_segment=True)  # (batch, N, T', d_model)
            x = self.g_att(x)
            x = self.feed_forward_gcn(x)
            return x  # output:  (batch, N, T', d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        """
        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, adj, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):

        # 动态GCN输出(B, N, T, D)--> 图注意的输入(B, T, N, D)
        x = x.permute(0, 2, 1, 3)

        Wx = torch.matmul(x, self.W)
        Wx1 = torch.matmul(Wx, self.a[:self.out_features, :])
        Wx2 = torch.matmul(Wx, self.a[self.out_features:, :])

        # broadcast add
        e = Wx1 + Wx2.permute(0, 1, 3, 2)
        e = self.leakyrelu(e)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, Wx).permute(0, 2, 1, 3)  # 图注意的输出(B, T, N, D)--> 输出(B, N, T, D)
        if self.concat:
            return F.elu(out)
        else:
            return out


def search_index(max_len, num_of_depend, num_for_predict, points_per_hour, units):
    """
    Parameters
    ----------
    max_len: int, length of all encoder input 所有编码器输入的长度
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample 将预测每个样本的点数
    units: int, week: 7 * 24, day: 24, recent(hour): 1   时间单位的系数
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    """
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, distance_mx, nb_head,
               num_of_weeks, num_of_days, num_of_hours, points_per_hour, num_for_predict, alpha=0.1, dropout=.0,
               aware_temporal_context=True, ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0,
               residual_connection=True,
               use_LayerNorm=True):
    # LR rate means: graph Laplacian Regularization

    c = copy.deepcopy

    # 通过邻接矩阵，构造归一化的拉普拉斯矩阵
    # 这个归一化的邻接矩阵将在模型的空间注意力机制中使用
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)

    # 构造高斯矩阵
    gaussian_Adj_matrix = torch.FloatTensor(gaussian_Adj(distance_mx)).to(DEVICE)

    num_of_vertices = norm_Adj_matrix.shape[0]

    # geo_mask, sem_mask = mask_matrix(sh_mx, dtw_matrix, num_of_vertices, far_mask_delta=7, dtw_delta=5)
    # geo_mask, sem_mask = geo_mask.to(DEVICE), sem_mask.to(DEVICE)

    src_dense = nn.Linear(encoder_input_size, d_model)

    if ScaledSAt:  # employ spatial self attention  运用空间自我注意
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model),
                                                       dropout=dropout)
    else:  # 普通卷积
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection  目标输入投影

    # encoder temporal position embedding    编码器时间位置嵌入
    # 计算输入序列的最大长度，根据时间序列中的周数、天数和小时数以及预测时间步数等信息计算得出。
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict,
                  num_of_hours * num_for_predict)

    # 根据最大长度和时间信息计算编码器的时间位置编码
    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7 * 24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    print('TemporalPositionalEncoding max_len:', max_len)
    print('w_index:', w_index)
    print('d_index:', d_index)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    if aware_temporal_context:  # employ temporal trend-aware attention  运用时间趋势感知注意力
        # 用于编码器的空间自我注意力
        attn_ss = Local_Global_Attention_en(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour,
                                            kernel_size, global_heads_ratio=0.5, dropout=dropout)

        # 用于解码器的空间-时间自我注意力
        attn_st = Local_Global_Attention_de1(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
                                             kernel_size, global_heads_ratio=0.5,
                                             dropout=dropout)  # 使用一维卷积的查询自注意力和空间自注意力，用于编码器的空间-时间自注意力

        # 用于解码器的时间自我注意力
        # att_tt = Local_Global_Attention_de(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, num_for_predict,
        #                 kernel_size, global_heads_ratio=0.5, dropout=dropout)  # decoder的trend-aware attention用因果卷积的查询自注意力和键自注意力，用于解码器的

    else:  # employ traditional self attention  运用传统的自我关注
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        # att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len,
                                                              en_lookup_index)  # decoder temporal position embedding  解码器时间位置嵌入
        # decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        # decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        # decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and TE:
        encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len,
                                                              en_lookup_index)  # decoder temporal position embedding  解码器时间位置嵌入
        # decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        # decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        # decoder_embedding = nn.Sequential(trg_dense)

    # gat = GAT(d_model, gaussian_Adj_matrix, dropout, alpha)  # 多层GAT
    gat = GraphAttentionLayer(d_model, d_model, gaussian_Adj_matrix, dropout, alpha)  # 单层

    encoderLayer = EncoderLayer(d_model, attn_ss, gat, c(position_wise_gcn), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, attn_st, gat, c(position_wise_gcn), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

    decoder = Decoder(decoderLayer, num_layers)

    fc = nn.Linear(d_model, decoder_output_size)

    model = EncoderDecoder(encoder,
                           decoder,
                           encoder_embedding,
                           fc,
                           DEVICE)
    # param init  对模型的参数进行初始化
    for p in model.parameters():
        if p.dim() > 1:  # 只有具有多个维度的参数才会被初始化
            nn.init.xavier_uniform_(p)

    return model
