import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    import timm
except ImportError:
    timm = None

class MultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with dropout and layer norm"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention with dropout
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        output = self.out(context)
        output = self.proj_dropout(output)
        return output, attention


class MHAVH(nn.Module):
    """Enhanced Multi-Head Attention Vision Hybrid model with improved architecture"""
    
    def __init__(self, num_classes=4, seq_length=30, feature_dim=512, backbone='resnet50', dropout=0.3):
        super().__init__()
        
        # Use better backbone (ResNet50 or EfficientNet)
        if timm and backbone.startswith('efficientnet'):
            try:
                self.backbone_model = timm.create_model(backbone, pretrained=True, num_classes=0)
                backbone_feature_dim = self.backbone_model.num_features
            except:
                resnet = models.resnet50(pretrained=True)
                self.backbone_model = nn.Sequential(*list(resnet.children())[:-1])
                backbone_feature_dim = 2048
        else:
            # Use ResNet50 for better feature extraction
            resnet = models.resnet50(pretrained=True)
            self.backbone_model = nn.Sequential(*list(resnet.children())[:-1])
            backbone_feature_dim = 2048
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_dim = feature_dim
        self.dropout_rate = dropout
        
        # Learnable positional encoding with sinusoidal initialization
        self.positional_encoding = nn.Parameter(
            self._get_sinusoidal_encoding(seq_length, self.feature_dim)
        )
        
        # Temporal LSTM for better sequence modeling
        self.temporal_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.feature_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        
        # Multi-Head Attention layers
        self.mha1 = MultiHeadAttention(self.feature_dim, num_heads=8, dropout=dropout)
        self.mha2 = MultiHeadAttention(self.feature_dim, num_heads=8, dropout=dropout)
        self.mha3 = MultiHeadAttention(self.feature_dim, num_heads=8, dropout=dropout)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.feature_dim)
        self.ln2 = nn.LayerNorm(self.feature_dim)
        self.ln3 = nn.LayerNorm(self.feature_dim)
        self.ln_lstm = nn.LayerNorm(self.feature_dim)
        
        # Enhanced Feed-forward network with GELU
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 4, self.feature_dim),
            nn.Dropout(dropout)
        )
        
        # Attention pooling for better aggregation
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def _get_sinusoidal_encoding(self, seq_length, d_model):
        """Generate sinusoidal positional encoding"""
        position = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        encoding = torch.zeros(1, seq_length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        
        return encoding
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, height, width)
        
        Returns:
            output: (batch, num_classes)
            attentions: tuple of attention weights
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features efficiently by processing all frames at once
        x_flat = x.view(batch_size * seq_len, c, h, w)
        features_flat = self.backbone_model(x_flat)
        
        if len(features_flat.shape) > 2:
            features_flat = features_flat.view(features_flat.size(0), -1)
        
        # Project features
        features = self.feature_projection(features_flat)
        features = features.view(batch_size, seq_len, self.feature_dim)
        
        # Add positional encoding
        features = features + self.positional_encoding[:, :seq_len, :]
        
        # Temporal LSTM processing
        lstm_out, _ = self.temporal_lstm(features)
        features = self.ln_lstm(features + lstm_out)
        
        # First multi-head attention block with residual connection
        attn_out1, attention1 = self.mha1(features)
        features = self.ln1(features + attn_out1)
        
        # Second multi-head attention block with residual connection
        attn_out2, attention2 = self.mha2(features)
        features = self.ln2(features + attn_out2)
        
        # Third multi-head attention block
        attn_out3, attention3 = self.mha3(features)
        features = self.ln3(features + attn_out3)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(features)
        features = features + ffn_out
        
        # Attention-based pooling (better than average pooling)
        attn_weights = self.attention_pooling(features)
        pooled = torch.sum(features * attn_weights, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, (attention1, attention2, attention3)