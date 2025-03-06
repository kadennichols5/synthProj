import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class VisualEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 latent_dim: int = 64):
        super().__init__()
        
        # Input dimensions
        self.time_dim = 1
        self.stem_dim = 4  # vocals, drums, bass, other
        self.spectral_dim = input_dim - self.time_dim - self.stem_dim
        
        # Validate dimensions of spectral data, must be positive
        if self.spectral_dim <= 0:
            raise ValueError(f"Input dimension ({input_dim}) must be greater than {self.time_dim + self.stem_dim} "
                            f"to accommodate time ({self.time_dim}) and stem ({self.stem_dim}) dimensions")
        
        # Output dimensions
        self.shape_dim = 6 # shape category output
        self.motion_dim = 6 # velocity and acceleration output
        self.texture_dim = 8 # texture parameters output
        self.color_dim = 4 # color parameters output - RGBA
        self.brightness_dim = 1 # brightness scalar output
        self.position_dim = 3 # position output - XYZ
        self.pattern_dim = 6 # pattern category output
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim), # fully connected layer
                nn.LayerNorm(hidden_dim), # normalize activations for stable training
                nn.LeakyReLU(), # leaky relu activation function (helps with gradient flow)
                nn.Dropout(0.2) # dropout layer to prevent overfitting (ie randomly drop out 20% of neurons)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers) # sequential container of layers
        
        # Latent space
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim) # fully connected layer to reduce dimensionality
        
        # Decoder heads for each visual parameter
        self.shape_head = nn.Sequential(
            nn.Linear(latent_dim, self.shape_dim), # fully connected layer to output shape categories
            nn.Softmax(dim=-1)  # For shape categories
        )
        
        self.motion_head = nn.Sequential(
            nn.Linear(latent_dim, self.motion_dim), # fully connected layer to output velocity and acceleration
            nn.Tanh()  # For velocity and acceleration
        )
        
        self.texture_head = nn.Sequential(
            nn.Linear(latent_dim, self.texture_dim), # fully connected layer to output texture parameters
            nn.Sigmoid()  # For texture parameters
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(latent_dim, self.color_dim), # fully connected layer to output color parameters
            nn.Sigmoid()  # For RGBA values
        )
        
        self.brightness_head = nn.Sequential(
            nn.Linear(latent_dim, self.brightness_dim), # fully connected layer to output brightness scalar
            nn.Sigmoid()  # For brightness scalar
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(latent_dim, self.position_dim), # fully connected layer to output position
            nn.Tanh()  # For XYZ coordinates
        )
        
        self.pattern_head = nn.Sequential(
            nn.Linear(latent_dim, self.pattern_dim), # fully connected layer to output pattern categories       
            nn.Softmax(dim=-1)  # For pattern categories
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode input
        encoded = self.encoder(x) # pass input through encoder
        latent = self.fc_latent(encoded) # reduce dimensionality of encoded input   
        
        # Generate visual parameters - output of each head is a tensor of shape (batch_size, 1)
        return {
            'shape': self.shape_head(latent),
            'motion': self.motion_head(latent),
            'texture': self.texture_head(latent),
            'color': self.color_head(latent),
            'brightness': self.brightness_head(latent),
            'position': self.position_head(latent),
            'pattern': self.pattern_head(latent)
        }

    def loss_function(self, 
                     outputs: Dict[str, torch.Tensor], 
                     target: Dict[str, torch.Tensor],
                     weights: Dict[str, float] = None) -> torch.Tensor:
        """
        Custom loss function combining multiple objectives

        Cross entropy loss for shape, pattern and color (categorical)
        MSE loss for motion, brightness and position (continuous)
        """
        if weights is None:
            weights = {
                'shape': 1.0,
                'motion': 1.0,
                'texture': 1.0,
                'color': 1.0,
                'brightness': 1.0,
                'position': 1.0,
                'pattern': 1.0
            }
            
        losses = {
            'shape': F.cross_entropy(outputs['shape'], target['shape']),
            'motion': F.mse_loss(outputs['motion'], target['motion']),
            'texture': F.mse_loss(outputs['texture'], target['texture']),
            'color': F.mse_loss(outputs['color'], target['color']),
            'brightness': F.mse_loss(outputs['brightness'], target['brightness']),
            'position': F.mse_loss(outputs['position'], target['position']),
            'pattern': F.cross_entropy(outputs['pattern'], target['pattern'])
        }
        
        total_loss = sum(weights[k] * losses[k] for k in losses)
        return total_loss

class UnsupervisedVisualEncoder(VisualEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add consistency loss components
        self.temporal_consistency = nn.MSELoss()
        self.spectral_consistency = nn.MSELoss()
        
    def unsupervised_loss(self, 
                         outputs: Dict[str, torch.Tensor],
                         input_sequence: torch.Tensor,
                         weights: Dict[str, float] = None) -> torch.Tensor:
        """
        Unsupervised loss based on:
        1. Temporal consistency
        2. Spectral energy correlation
        3. Perceptual smoothness
        4. Physical constraints
        """
        if weights is None:
            weights = {
                'temporal': 1.0,
                'spectral': 0.5,
                'smoothness': 0.3,
                'physics': 0.2
            }
            
        # Temporal consistency loss
        temporal_loss = self.temporal_consistency_loss(outputs)
        
        # Spectral correlation loss
        spectral_loss = self.spectral_correlation_loss(outputs, input_sequence)
        
        # Smoothness loss
        smoothness_loss = self.smoothness_loss(outputs)
        
        # Physical constraints loss
        physics_loss = self.physics_constraints_loss(outputs)
        
        total_loss = (
            weights['temporal'] * temporal_loss +
            weights['spectral'] * spectral_loss +
            weights['smoothness'] * smoothness_loss +
            weights['physics'] * physics_loss
        )
        
        return total_loss
