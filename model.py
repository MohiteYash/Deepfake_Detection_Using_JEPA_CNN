"""
CNN-JEPA Model Architecture for Deepfake Detection

Components:
1. CNN Encoder (ResNet-based or custom)
2. Context Encoder
3. Target Encoder (EMA updated)
4. Predictor Network
5. Spatial Attention Module (novel component)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class CNNEncoder(nn.Module):
    """
    CNN-based encoder for extracting patch embeddings
    Uses ResNet backbone or custom CNN
    """
    
    def __init__(self, encoder_type='resnet18', embedding_dim=512, pretrained=True):
        """
        Args:
            encoder_type: Type of encoder ('resnet18', 'resnet34', 'custom_cnn')
            embedding_dim: Dimension of output embeddings
            pretrained: Use pretrained weights
        """
        super(CNNEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        
        if encoder_type == 'resnet18':
            # Use ResNet18 as backbone
            resnet = models.resnet18(pretrained=pretrained)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
            
        elif encoder_type == 'resnet34':
            # Use ResNet34 as backbone
            resnet = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
            
        elif encoder_type == 'custom_cnn':
            # Custom CNN architecture
            self.backbone = self._build_custom_cnn()
            backbone_dim = 512
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Projection head to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def _build_custom_cnn(self):
        """Build custom CNN architecture"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W] or [B, N, C, H, W] for batch of patches
            
        Returns:
            embeddings: Output embeddings [B, embedding_dim] or [B, N, embedding_dim]
        """
        # Handle batch of patches
        if x.dim() == 5:  # [B, N, C, H, W]
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            
            # Extract features
            features = self.backbone(x)  # [B*N, backbone_dim, 1, 1]
            features = features.view(B * N, -1)  # [B*N, backbone_dim]
            
            # Project to embedding space
            embeddings = self.projection(features)  # [B*N, embedding_dim]
            embeddings = embeddings.view(B, N, -1)  # [B, N, embedding_dim]
            
        else:  # [B, C, H, W]
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            embeddings = self.projection(features)
        
        return embeddings


class SpatialAttention(nn.Module):
    """
    Spatial attention module for feature refinement
    Helps model focus on important regions
    """
    
    def __init__(self, embedding_dim, num_heads=8):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_heads: Number of attention heads
        """
        super(SpatialAttention, self).__init__()
        
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings [B, N, embedding_dim]
            
        Returns:
            attended: Attended embeddings [B, N, embedding_dim]
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, H, N, d]
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        
        # Output projection
        attended = self.out_proj(attended)
        
        # Residual connection + layer norm
        output = self.norm(x + attended)
        
        return output


class ContextEncoder(nn.Module):
    """
    Context encoder that processes context patches
    Aggregates information from visible patches
    """
    
    def __init__(self, embedding_dim, hidden_dim, use_attention=True, num_heads=8):
        """
        Args:
            embedding_dim: Dimension of patch embeddings
            hidden_dim: Hidden dimension for processing
            use_attention: Whether to use spatial attention
            num_heads: Number of attention heads
        """
        super(ContextEncoder, self).__init__()
        
        self.use_attention = use_attention
        
        # Attention module (optional)
        if use_attention:
            self.attention = SpatialAttention(embedding_dim, num_heads)
        
        # MLP for context processing
        self.context_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, context_embeddings):
        """
        Args:
            context_embeddings: Context patch embeddings [B, N_context, embedding_dim]
            
        Returns:
            processed_context: Processed context [B, N_context, embedding_dim]
        """
        # Apply attention if enabled
        if self.use_attention:
            context_embeddings = self.attention(context_embeddings)
        
        # Process through MLP
        processed_context = self.context_mlp(context_embeddings)
        
        return processed_context


class PredictorNetwork(nn.Module):
    """
    Predictor network that predicts target embeddings from context
    Uses MLP with multiple layers
    """
    
    def __init__(self, embedding_dim, hidden_dim, depth=3):
        """
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension
            depth: Number of layers
        """
        super(PredictorNetwork, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, context_embeddings, target_positions):
        """
        Predict target embeddings from context
        
        Args:
            context_embeddings: Context embeddings [B, N_context, embedding_dim]
            target_positions: Number of target patches to predict
            
        Returns:
            predictions: Predicted target embeddings [B, N_target, embedding_dim]
        """
        B, N_context, D = context_embeddings.shape
        
        # Aggregate context (mean pooling)
        aggregated_context = context_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Expand to target positions
        aggregated_context = aggregated_context.expand(B, target_positions, D)  # [B, N_target, D]
        
        # Predict target embeddings
        predictions = self.predictor(aggregated_context)
        
        return predictions


class CNNJEPA(nn.Module):
    """
    Complete CNN-JEPA model for deepfake detection
    
    Architecture:
    1. CNN Encoder - extracts patch embeddings
    2. Context Encoder - processes context patches
    3. Target Encoder - EMA-updated encoder for target patches
    4. Predictor - predicts target embeddings from context
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object
        """
        super(CNNJEPA, self).__init__()
        
        self.config = config
        
        # CNN Encoder (shared)
        self.encoder = CNNEncoder(
            encoder_type=config.ENCODER_TYPE,
            embedding_dim=config.EMBEDDING_DIM,
            pretrained=True
        )
        
        # Context Encoder
        self.context_encoder = ContextEncoder(
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.CONTEXT_HIDDEN_DIM,
            use_attention=config.USE_ATTENTION,
            num_heads=config.ATTENTION_HEADS
        )
        
        # Target Encoder (EMA copy of encoder)
        self.target_encoder = CNNEncoder(
            encoder_type=config.ENCODER_TYPE,
            embedding_dim=config.EMBEDDING_DIM,
            pretrained=True
        )
        
        # Initialize target encoder with same weights as encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Freeze target encoder (will be updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor Network
        self.predictor = PredictorNetwork(
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.PREDICTOR_HIDDEN_DIM,
            depth=config.PREDICTOR_DEPTH
        )
        
    def forward(self, patches, context_indices, target_indices):
        """
        Forward pass for JEPA training
        
        Args:
            patches: All patches [B, N, C, H, W]
            context_indices: Indices of context patches [B, N_context]
            target_indices: Indices of target patches [B, N_target]
            
        Returns:
            predictions: Predicted target embeddings [B, N_target, embedding_dim]
            targets: Actual target embeddings [B, N_target, embedding_dim]
            context_embeddings: Context embeddings [B, N_context, embedding_dim]
        """
        B, N, C, H, W = patches.shape
        
        # Extract context patches
        context_patches = torch.stack([
            patches[b, context_indices[b]] for b in range(B)
        ], dim=0)  # [B, N_context, C, H, W]
        
        # Extract target patches
        target_patches = torch.stack([
            patches[b, target_indices[b]] for b in range(B)
        ], dim=0)  # [B, N_target, C, H, W]
        
        # Encode context patches
        context_embeddings = self.encoder(context_patches)  # [B, N_context, embedding_dim]
        
        # Process context
        processed_context = self.context_encoder(context_embeddings)  # [B, N_context, embedding_dim]
        
        # Predict target embeddings
        N_target = target_patches.shape[1]
        predictions = self.predictor(processed_context, N_target)  # [B, N_target, embedding_dim]
        
        # Encode target patches (with EMA encoder, no gradients)
        with torch.no_grad():
            targets = self.target_encoder(target_patches)  # [B, N_target, embedding_dim]
        
        return predictions, targets, context_embeddings
    
    def encode_patches(self, patches):
        """
        Encode all patches (used for anomaly detection)
        
        Args:
            patches: All patches [B, N, C, H, W]
            
        Returns:
            embeddings: Patch embeddings [B, N, embedding_dim]
        """
        return self.encoder(patches)
    
    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """
        Update target encoder using EMA (Exponential Moving Average)
        
        Args:
            momentum: EMA momentum coefficient
        """
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)


if __name__ == "__main__":
    # Test model
    from config import Config
    
    print("Testing CNN-JEPA model...")
    
    model = CNNJEPA(Config).to(Config.DEVICE)
    print(f"Model created on {Config.DEVICE}")
    
    # Create dummy data
    B, N, C, H, W = 2, 16, 3, 56, 56  # 2 images, 16 patches each
    patches = torch.randn(B, N, C, H, W).to(Config.DEVICE)
    context_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(Config.DEVICE)
    target_indices = torch.tensor([[10, 11, 12, 13, 14, 15], 
                                   [10, 11, 12, 13, 14, 15]]).to(Config.DEVICE)
    
    # Forward pass
    predictions, targets, context_embeddings = model(patches, context_indices, target_indices)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Context embeddings shape: {context_embeddings.shape}")
    
    # Test EMA update
    model.update_target_encoder(Config.EMA_MOMENTUM)
    print("\nEMA update successful")
