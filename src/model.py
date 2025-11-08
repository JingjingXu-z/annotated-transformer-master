import torch
import torch.nn as nn
import math
# 确保导入路径正确（根据之前的配置，用绝对导入）
from config import Config
# 导入utils中的tokenizer，获取真实词汇表大小
from utils import tokenizer

config = Config()

# ===================== 基础绝对位置编码（稳定无维度问题）=====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        # 初始化位置编码矩阵 [max_len, d_model]，适配句子最大长度50
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 位置编码公式：sin(pos/(10000^(2i/d_model)))、cos(pos/(10000^(2i/d_model)))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状：[batch_size, seq_len, d_model]，添加位置编码后返回
        x = x + self.pe[:x.size(1), :]
        return x

# ===================== 多头自注意力（移除相对位置编码，确保稳定）=====================
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_k = config.d_model // config.n_heads  # 128//4=32，每个头的维度
        # Q/K/V线性投影层
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # 1. 线性投影 + 拆分多头：[batch, seq_len, d_model] → [batch, n_heads, seq_len, d_k]
        q_proj = self.w_q(q).view(batch_size, -1, config.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, config.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, config.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数：[batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. 应用掩码（避免padding和未来token干扰）
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len] → 广播适配多头
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. 计算权重与输出
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_proj)  # [batch, n_heads, seq_len, d_k]

        # 5. 拼接多头 + 输出投影：[batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, config.d_model)
        attn_output = self.w_o(attn_output)

        # 6. 残差连接 + 层归一化
        output = self.layer_norm(q + attn_output)
        return output, attn_weights

# ===================== 位置感知前馈网络（保持原逻辑）=====================
class PositionWiseFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        ffn_output = self.fc2(self.relu(self.fc1(x)))
        output = self.layer_norm(x + ffn_output)
        return output

# ===================== Encoder层（添加绝对位置编码）=====================
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = PositionWiseFFN()

    def forward(self, x, padding_mask):
        attn_output, _ = self.self_attn(x, x, x, padding_mask)
        ffn_output = self.ffn(attn_output)
        return ffn_output

# ===================== Decoder层（保持原逻辑）=====================
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.ffn = PositionWiseFFN()

    def forward(self, x, enc_output, padding_mask, future_mask):
        masked_attn_output, _ = self.masked_self_attn(x, x, x, mask=future_mask)
        cross_attn_output, _ = self.cross_attn(q=masked_attn_output, k=enc_output, v=enc_output, mask=padding_mask)
        ffn_output = self.ffn(cross_attn_output)
        return ffn_output

# ===================== Encoder（修正词典大小+添加位置编码）=====================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 词典大小 = 离线Tokenizer的词汇表大小（51），不再硬编码32000
        self.vocab_size = len(tokenizer.word2id)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=config.d_model)
        self.pos_enc = PositionalEncoding(d_model=config.d_model)  # 添加绝对位置编码
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config.n_encoder_layers)])

    def forward(self, x, padding_mask):
        seq_len = x.shape[1]
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        embed = self.embedding(x) * math.sqrt(config.d_model)
        embed = self.pos_enc(embed)  # 加入位置编码
        embed = self.dropout(embed)

        enc_output = embed
        for layer in self.layers:
            enc_output = layer(enc_output, padding_mask)
        return enc_output

# ===================== Decoder（修正词典大小+添加位置编码）=====================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = len(tokenizer.word2id)  # 与Encoder一致
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=config.d_model)
        self.pos_enc = PositionalEncoding(d_model=config.d_model)  # 添加绝对位置编码
        self.dropout = nn.Dropout(p=0.1)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config.n_decoder_layers)])
        self.fc_out = nn.Linear(config.d_model, self.vocab_size)  # 输出维度=词典大小

    def forward(self, x, enc_output, padding_mask, future_mask):
        seq_len = x.shape[1]
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        embed = self.embedding(x) * math.sqrt(config.d_model)
        embed = self.pos_enc(embed)  # 加入位置编码
        embed = self.dropout(embed)

        dec_output = embed
        for layer in self.layers:
            dec_output = layer(dec_output, enc_output, padding_mask, future_mask)
        
        logits = self.fc_out(dec_output)  # [batch, seq_len-1, vocab_size]
        return logits

# ===================== 完整Transformer（保持原逻辑）=====================
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt):
        # 1. 生成Encoder的padding掩码（不变）
        from utils import create_padding_mask, create_future_mask
        padding_mask = create_padding_mask(src)  # [batch, 1, seq_len_src]

        # 2. 提取Decoder的实际输入（tgt[:, :-1]，长度=原长度-1）
        tgt_input = tgt[:, :-1]  # 例如：原长度50 → 49

        # 3. 基于Decoder实际输入生成future_mask（关键修复！）
        future_mask = create_future_mask(tgt_input)  # 长度49，与tgt_input匹配

        # 4. Encoder前向传播（不变）
        enc_output = self.encoder(src, padding_mask)

        # 5. Decoder前向传播（使用tgt_input和匹配的future_mask）
        dec_logits = self.decoder(tgt_input, enc_output, padding_mask, future_mask)
        return dec_logits