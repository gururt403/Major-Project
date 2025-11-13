import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
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
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        output = self.out(context)
        return output, attention


class MHAVH(nn.Module):
    """Multi-Head Attention Vision Hybrid model for posture classification"""
    
    def __init__(self, num_classes=4, seq_length=30, feature_dim=512):
        super().__init__()
        
        # CNN backbone (ResNet18) for spatial feature extraction
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze early layers
        for i, param in enumerate(self.feature_extractor.parameters()):
            if i < 40:  # Freeze first 40 layers
                param.requires_grad = False
        
        self.feature_dim = feature_dim
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, seq_length, self.feature_dim)
        )
        
        # Multi-Head Attention layers
        self.mha1 = MultiHeadAttention(self.feature_dim, num_heads=8)
        self.mha2 = MultiHeadAttention(self.feature_dim, num_heads=8)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.feature_dim)
        self.ln2 = nn.LayerNorm(self.feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim * 4, self.feature_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, height, width)
        
        Returns:
            output: (batch, num_classes)
            attentions: tuple of attention weights
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract features from each frame
        features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.feature_extractor(frame)
            feat = feat.view(batch_size, -1)
            features.append(feat)
        
        # Stack features: (batch, seq_len, feature_dim)
        features = torch.stack(features, dim=1)
        
        # Add positional encoding
        features = features + self.positional_encoding[:, :seq_len, :]
        
        # First multi-head attention block with residual connection
        attn_out1, attention1 = self.mha1(features)
        features = self.ln1(features + attn_out1)
        
        # Second multi-head attention block with residual connection
        attn_out2, attention2 = self.mha2(features)
        features = self.ln2(features + attn_out2)
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(features)
        features = features + ffn_out
        
        # Global average pooling over sequence
        pooled = torch.mean(features, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, (attention1, attention2)