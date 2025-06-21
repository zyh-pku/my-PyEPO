from torch import nn
import torch
# torch.manual_seed(42)  


class EnhancedLinearRegression(nn.Module):
    """
    Enhanced linear regression model with batch normalization
    for improved training stability
    """
    def __init__(self, k: int, dropout_rate=0.0):
        super().__init__()
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(k)
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # Linear layer - maps k features to 1 output
        self.linear = nn.Linear(in_features=k, out_features=1)
        
        # Initialize weights properly
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (batch_size, N, k) -> reshape for batch norm
        batch_size, N, k = x.shape
        x_reshaped = x.reshape(-1, k)  # (batch_size*N, k)
        
        # Apply batch normalization
        x_normalized = self.batch_norm(x_reshaped)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)
            
        # Apply linear layer
        output = self.linear(x_normalized)  # (batch_size*N, 1)
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, N)  # (batch_size, N)
        
        return output
    
    

class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP model with batch normalization and optional dropout
    """
    def __init__(self, k: int, hidden_dim: int=32, dropout_rate=0.0):
        super().__init__()
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(k)
        
        # First layer: k -> hidden_dim
        self.fc1 = nn.Linear(k, hidden_dim)
        
        # Second layer: hidden_dim -> 1
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights properly
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # x: (batch_size, N, k) -> reshape for batch norm
        batch_size, N, k = x.shape
        x_reshaped = x.reshape(-1, k) # (batch_size*N, k)
        
        # Apply batch normalization
        x_normalized = self.batch_norm(x_reshaped)
        
        # First linear layer + activation
        x_hidden = self.fc1(x_normalized)
        x_hidden = self.activation(x_hidden)
        
        # Optional dropout 
        if self.dropout is not None:
            x_hidden = self.dropout(x_hidden)
        
        # Second linear layer (output)
        output = self.fc2(x_hidden) # (batch_size*N, 1)
        
        # Reshape back to (batch_size, N)
        output = output.reshape(batch_size, N)
        
        return output
    
class TwoLayerLSTM(nn.Module):
    """
    更高效的 LSTM 实现，处理时间序列数据
    """
    def __init__(self, k, hidden_dim=32, lstm_hidden_dim=64, dropout_rate=0.0):
        super(TwoLayerLSTM, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM 层
        self.lstm = nn.LSTM(k, lstm_hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        
        # 重塑为 LSTM 输入: (batch_size*N, lookback, k)
        x_lstm = x.reshape(-1, lookback, k)
        
        # LSTM 处理
        lstm_out, _ = self.lstm(x_lstm)
        # 使用最后一个时间步的输出
        lstm_final = lstm_out[:, -1, :]  # shape: (batch_size*N, lstm_hidden_dim)
        
        # 全连接层处理
        output = self.fc_layers(lstm_final)  # shape: (batch_size*N, 1)
        
        # 重塑为最终输出
        output = output.view(batch_size, N)  # shape: (batch_size, N)
        
        return output






import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. TCN (Temporal Convolutional Network) - 推荐首选
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate=0.1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]] if self.conv2.padding[0] > 0 else out
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)

class TCNModel(nn.Module):
    """TCN模型 - 解决梯度问题的最佳选择"""
    def __init__(self, k, hidden_dim=64, num_layers=4, kernel_size=3, dropout_rate=0.1):
        super(TCNModel, self).__init__()
        self.k = k
        
        # TCN layers with increasing dilation
        self.tcn_layers = nn.ModuleList()
        in_channels = k
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(TCNBlock(in_channels, hidden_dim, kernel_size, dilation, dropout_rate))
            in_channels = hidden_dim
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        
        # Reshape for TCN: (batch_size*N, k, lookback)
        x = x.reshape(-1, lookback, k).transpose(1, 2)
        
        # TCN processing
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Output
        output = self.output_layer(x)  # (batch_size*N, 1)
        
        return output.view(batch_size, N)


# 2. 轻量级Transformer - 适用于中等长度序列
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LightTransformer(nn.Module):
    """轻量级Transformer - 适合中等长度序列"""
    def __init__(self, k, d_model=64, nhead=4, num_layers=2, dropout_rate=0.1):
        super(LightTransformer, self).__init__()
        self.k = k
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(k, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model//2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        
        # Reshape: (batch_size*N, lookback, k)
        x = x.reshape(-1, lookback, k)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch_size*N, lookback, d_model)
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)  # (batch_size*N, lookback, d_model)
        
        # Global average pooling + output
        x = x.mean(dim=1)  # (batch_size*N, d_model)
        output = self.output_layer(x)  # (batch_size*N, 1)
        
        return output.view(batch_size, N)


# 3. DLinear - 极简但有效的线性模型
class DLinear(nn.Module):
    """DLinear - 分解线性模型，性能出奇地好"""
    def __init__(self, k, lookback, individual=False):
        super(DLinear, self).__init__()
        self.lookback = lookback
        self.individual = individual
        self.k = k
        
        if individual:
            # 每个特征独立的线性层
            self.linear_seasonal = nn.ModuleList([nn.Linear(lookback, 1) for _ in range(k)])
            self.linear_trend = nn.ModuleList([nn.Linear(lookback, 1) for _ in range(k)])
        else:
            # 共享线性层
            self.linear_seasonal = nn.Linear(lookback * k, 1)
            self.linear_trend = nn.Linear(lookback * k, 1)
    
    def series_decomp(self, x):
        # 简单的移动平均分解
        kernel_size = 25
        if kernel_size >= x.size(-1):
            kernel_size = x.size(-1) // 2
        
        # 移动平均作为趋势
        trend = F.avg_pool1d(x.transpose(-1, -2), kernel_size=kernel_size, 
                           stride=1, padding=kernel_size//2).transpose(-1, -2)
        
        # 残差作为季节性
        seasonal = x - trend
        return seasonal, trend
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        x = x.reshape(-1, lookback, k)  # (batch_size*N, lookback, k)
        
        # 分解
        seasonal, trend = self.series_decomp(x)
        
        if self.individual:
            seasonal_output = torch.zeros(x.size(0), 1, device=x.device)
            trend_output = torch.zeros(x.size(0), 1, device=x.device)
            
            for i in range(k):
                seasonal_output += self.linear_seasonal[i](seasonal[:, :, i])
                trend_output += self.linear_trend[i](trend[:, :, i])
            
            output = seasonal_output + trend_output
        else:
            # Flatten
            seasonal_flat = seasonal.reshape(x.size(0), -1)
            trend_flat = trend.reshape(x.size(0), -1)
            
            seasonal_output = self.linear_seasonal(seasonal_flat)
            trend_output = self.linear_trend(trend_flat)
            output = seasonal_output + trend_output
        
        return output.view(batch_size, N)


# 4. 混合CNN-Attention模型
class CNNAttentionModel(nn.Module):
    """CNN + Self-Attention混合模型"""
    def __init__(self, k, hidden_dim=64, num_heads=4, dropout_rate=0.1):
        super(CNNAttentionModel, self).__init__()
        self.k = k
        
        # 1D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(k, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, N, lookback, k)
        batch_size, N, lookback, k = x.shape
        
        # Reshape for CNN: (batch_size*N, k, lookback)
        x = x.reshape(-1, lookback, k).transpose(1, 2)
        
        # CNN processing
        x = self.conv_layers(x)  # (batch_size*N, hidden_dim, lookback)
        
        # Transpose for attention: (batch_size*N, lookback, hidden_dim)
        x = x.transpose(1, 2)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        # Back to CNN format and output
        x = x.transpose(1, 2)  # (batch_size*N, hidden_dim, lookback)
        output = self.output_layer(x)
        
        return output.view(batch_size, N)
"""

# 使用示例
def create_model(model_type, k, **kwargs):
    if model_type == 'tcn':
        return TCNModel(k, **kwargs)
    elif model_type == 'transformer':
        return LightTransformer(k, **kwargs)
    elif model_type == 'dlinear':
        return DLinear(k, **kwargs)
    elif model_type == 'cnn_attention':
        return CNNAttentionModel(k, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# 示例用法
if __name__ == "__main__":
    k = 10  # 特征数
    lookback = 50  # 序列长度
    batch_size = 32
    N = 100
    
    # 创建不同模型
    models = {
        'TCN': create_model('tcn', k, hidden_dim=64),
        'Transformer': create_model('transformer', k, d_model=64),
        'DLinear': create_model('dlinear', k, lookback=lookback),
        'CNN-Attention': create_model('cnn_attention', k, hidden_dim=64)
    }
    
    # 测试输入
    x = torch.randn(batch_size, N, lookback, k)
    
    for name, model in models.items():
        output = model(x)
        print(f"{name}: {output.shape}")  # 应该都是 (batch_size, N)



"""        
    
