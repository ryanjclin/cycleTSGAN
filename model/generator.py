import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, source_encoding):
        super(PositionalEncoding, self).__init__()
        # 使用 source_encoding 的長度作為 max_len
        max_len = len(source_encoding)
        
        # 確保 max_len 是整數，並生成 positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
<<<<<<< HEAD
        # flatten
        pe.flatten()
        pe = pe.unsqueeze(0)
=======
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        
        # 保存生成的 positional encoding
>>>>>>> 6a2420677780346e91b30271a80e53b860d423b4
        self.register_buffer("pe", pe)
        
        # 保存自定義的 source_encoding
        self.source_encoding = torch.tensor(source_encoding, dtype=torch.long)

<<<<<<< HEAD
    def forward(self, x):
        print(self.pe.shape)
        print(x.shape)
        x = x + self.pe[:, : x.size(1), :]
=======
    def forward(self, x):  # [var_num, batch_size, d_model]
        # x 的 shape: [var_num, batch_size, d_model]
        # 使用 source_encoding 為每個變數取對應的 positional encoding
        var_num = x.size(0)
        batch_size = x.size(1)

        # 利用 source_encoding 來選擇對應的 position encoding
        pe_indices = self.source_encoding.unsqueeze(1).expand(-1, batch_size)
        custom_pe = self.pe[:, pe_indices, :].squeeze(0)  # shape: [var_num, batch_size, d_model]

        # 加上 positional encoding
        x = x + custom_pe
>>>>>>> 6a2420677780346e91b30271a80e53b860d423b4
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
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, source_encoding):
        super(MultiLayerTransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, source_encoding)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
<<<<<<< HEAD
    #     src = self.pos_encoder(src) # source encoding
=======
        src = self.pos_encoder(src) # we do do position encoding. [var_num, batch_size, d_model]
>>>>>>> 1db3eac32883ffc1853014597c8aa1d055eb3197
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)
        return src


# Generator Class with Multi-Layer Transformer Encoder
class Generator(nn.Module):
    def __init__(
        self,
        config,
        source_encoding,
    ):
        super(Generator, self).__init__()
        self.batch_size = config['batch_size']
        self.var_num = config['var_num']
        self.seq_len = config['seq_len']
        self.source_encoding = source_encoding
        self.num_layers = config['gen_num_layers']
        self.d_model = config['gen_d_model']
        self.nhead = config['gen_nhead']
        self.dim_feedforward = config['gen_dim_feedforward']
        self.dropout = config['gen_dropout']

        self.d_model = (self.d_model // self.nhead) * self.nhead

        # define multi-layer Transformer Encoder
        self.transformer_encoder = MultiLayerTransformerEncoder(
            num_layers = self.num_layers,
            d_model = self.d_model,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout,
            source_encoding = self.source_encoding,
        )

        # adjust input and output dimension
        self.input_fc = nn.Linear(self.seq_len, self.d_model)
        self.output_fc = nn.Linear(self.d_model, self.seq_len)

    def forward(self, x):
        """x : [batch_size, var_num, seq_len]"""

        # 1. adjust shape to be align with Transformer Encoder input shape: [batch_size, var_num, seq_len] -> [var_num, batch_size, seq_len]
        x = x.transpose(0, 1)

        # 2. use FC layer to change the dimension of "seq_len". [var_num, batch_size, d_model]
        x = self.input_fc(x)

        # 3. use multi-layer Transformer Encoder
        x = self.transformer_encoder(x)

        # 4. back to "seq_len" oringinal dimension
        x = self.output_fc(x)

        # 5. reverse step 1: [var_num, batch_size, seq_len] -> [batch_size, var_num, seq_len]
        x = x.transpose(0, 1)

        return x