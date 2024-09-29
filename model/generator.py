import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class Generator(nn.Module):
#     def __init__(self, batch_size, var_num, seq_len, dim_z):
#         super(Generator, self).__init__()
#         self.batch_size = batch_size
#         self.var_num = var_num
#         self.seq_len = seq_len
#         self.dim_z = dim_z

#         self.fc1 = nn.Linear(self.seq_len, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 16)
#         self.fc5 = nn.Linear(16, 32)
#         self.fc6 = nn.Linear(32, 64)
#         self.fc7 = nn.Linear(64, 128)
#         self.fc8 = nn.Linear(128, self.seq_len)

#     def forward(self, x):
#         '''x : [batch_size, var_num, seq_len]'''

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))
#         x = F.relu(self.fc7(x))
#         x = self.fc8(x)

#         return x


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


# single layer Transformer Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        assert d_model % nhead == 0, "d_model mod nhead has to be 0"
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# multi-layer Transformer Encoder
class MultiLayerTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super(MultiLayerTransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        # src = self.pos_encoder(src) # we do not do position encoding
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


# Generator Class with Multi-Layer Transformer Encoder
class Generator(nn.Module):
    def __init__(
        self,
        batch_size,
        var_num,
        seq_len,
        dim_z,
        num_layers = 4,
        d_model = 256 * 3,
        nhead = 3,
        dim_feedforward = 1024,
        dropout = 0.01,
        # num_layers = 6,
        # d_model = 512 * 4,
        # nhead = 4,
        # dim_feedforward = 512,
        # dropout = 0.01,
    ):
        assert d_model % nhead == 0, "d_model mod nhead has to be 0"

        d_model = (d_model // nhead) * nhead
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.var_num = var_num
        self.seq_len = seq_len
        self.dim_z = dim_z

        # define multi-layer Transformer Encoder
        self.transformer_encoder = MultiLayerTransformerEncoder(
            num_layers = num_layers,
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
        )

        # adjust input and output dimension
        self.input_fc = nn.Linear(self.seq_len, d_model)
        self.output_fc = nn.Linear(d_model, self.seq_len)

    def forward(self, x):
        """x : [batch_size, var_num, seq_len]"""

        # 1. adjust shape to be align with Transformer Encoder input shape: [batch_size, var_num, seq_len] -> [var_num, batch_size, seq_len]
        x = x.transpose(0, 1)

        # 2. use FC layer to change the dimension of "seq_len"
        x = self.input_fc(x)

        # 3. use multi-layer Transformer Encoder
        x = self.transformer_encoder(x)

        # 4. back to "seq_len" oringinal dimension
        x = self.output_fc(x)

        # 5. reverse step 1: [var_num, batch_size, seq_len] -> [batch_size, var_num, seq_len]
        x = x.transpose(0, 1)

        return x


# # 測試 Generator
# batch_size = 32
# var_num = 88
# seq_len = 40
# dim_z = 100
# generator = Generator(batch_size, var_num, seq_len, dim_z)

# # 測試數據
# input_tensor = torch.rand(batch_size, var_num, seq_len)
