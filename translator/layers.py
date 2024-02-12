import torch
import torch.nn as nn
import math

# Decoder del transformer
class EncoderLayer(nn.Module):
    def __init__(self, drop_out_rate=0.1):
        super().__init__()
        self.linear1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

# Encoder del transformer
class DecoderLayer(nn.Module):
    def __init__(self, drop_out_rate=0.1):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.masked_multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_3 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.multihead_attention(x_2, e_output, e_output, mask=e_mask)) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)

# Multihead attention para el transformer
class MultiheadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_k=64, drop_out_rate=0.1):
        super().__init__()
        self.inf = 1e9 # -INF
        self.num_heads = num_heads # numero de cabezas
        self.d_k = d_k # dimension de las cabezas
        self.d_model = d_model # dimension del modelo

        # W^Q, W^K, W^V en el articulo
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Self-attention dropout
        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Salida Final de la capa de atencion multiple
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape # (B, L, d_model)

        # Cálculo lineal + división en num_heads
        q = self.w_q(q).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, self.num_heads, self.d_k) # (B, L, num_heads, d_k)

        # Por conveniencia, convierta todos los tensores de tamaño (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Llevar a cabo self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2).contiguous().view(input_shape[0], -1, self.d_model) # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calcule puntuaciones de atención con atención de producto escalado
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # Si hay máscara, hacer puntos enmascarados -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax y multiplicación de K para calcular el valor de atención
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values

# Normalización de capas para el transformer   
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6, d_model=512):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)
        return x

# Feedforward layer para el transformer
class FeedFowardLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_out_rate=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x

# Posicional encoding para el transformer
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=512, seq_len=100, device='cpu'):
        super().__init__()
        self.d_model = d_model
        # Hacer una matriz de codificación posicional inicial con 0
        pe_matrix= torch.zeros(seq_len, self.d_model) # (L, d_model)

        # Calcular valores de codificación de posición
        for pos in range(seq_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / self.d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / self.d_model)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)
        return x