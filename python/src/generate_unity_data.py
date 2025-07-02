"""
Generate Unity-compatible visual data from trained model
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Add config to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "config"))
from paths import get_spectral_data_path, get_neural_output_path, get_project_info
from settings import settings

from neural_network import VisualEncoder, UnsupervisedVisualEncoder
from train_visual_encoder import SpectralDataset, custom_collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str, input_dim: int):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model with same architecture
    model = UnsupervisedVisualEncoder(input_dim=input_dim)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def generate_unity_data_from_model(model: nn.Module, 
                                  dataloader: DataLoader, 
                                  output_file: str = 'unity_visual_data.json',
                                  output_dir: str = None):
    """Generate Unity-compatible visual data from trained model."""
    if output_dir is None:
        output_dir = get_neural_output_path()
    output_dir = Path(output_dir)
    
    model.eval()
    unity_data = []
    
    logger.info("Generating Unity-compatible visual data...")
    
    with torch.no_grad():
        for batch_idx, spectral_data in enumerate(tqdm(dataloader, desc="Generating Unity data")):
            # Process each sample individually to get one set of parameters per audio segment
            for i in range(spectral_data.shape[0]):  # For each sample in batch
                # Get the first valid time step from the sequence
                sample_sequence = spectral_data[i]  # Shape: (sequence_length, features)
                
                # Find first non-zero row (actual data, not padding)
                non_zero_mask = torch.any(sample_sequence != 0, dim=1)
                if torch.any(non_zero_mask):
                    first_valid_idx = torch.where(non_zero_mask)[0][0]
                    # Extract just the first valid time step
                    single_timestep = sample_sequence[first_valid_idx:first_valid_idx+1]  # Shape: (1, features)
                    time_value = float(single_timestep[0, 0].cpu().numpy())
                else:
                    # If no valid data, use the first row
                    single_timestep = sample_sequence[0:1]  # Shape: (1, features)
                    time_value = 0.0
                
                # Process this single timestep through the model
                device = next(model.parameters()).device
                single_timestep = single_timestep.to(device)
                outputs = model(single_timestep)
                
                # Extract visual parameters for this sample
                # Each output is a tensor of shape (1, feature_dim)
                sample_data = {
                    'segment_id': f"batch_{batch_idx}_sample_{i}",
                    'time': time_value,
                    'visual_params': {
                        'shape': outputs['shape'][0].cpu().numpy().tolist(),
                        'motion': outputs['motion'][0].cpu().numpy().tolist(),
                        'texture': outputs['texture'][0].cpu().numpy().tolist(),
                        'color': outputs['color'][0].cpu().numpy().tolist(),
                        'brightness': float(outputs['brightness'][0].cpu().numpy()),  # This is a single value
                        'position': outputs['position'][0].cpu().numpy().tolist(),
                        'pattern': outputs['pattern'][0].cpu().numpy().tolist()
                    }
                }
                unity_data.append(sample_data)
    
    # Save Unity data
    unity_path = output_dir / output_file
    with open(unity_path, 'w') as f:
        json.dump(unity_data, f, indent=2)
    
    logger.info(f"Unity data saved to {unity_path}")
    logger.info(f"Generated {len(unity_data)} visual parameter sets")
    
    return unity_data

def main():
    """Main function to generate Unity data from trained model."""
    # Get paths
    spectral_data_path = get_spectral_data_path()
    neural_output_path = get_neural_output_path()
    
    logger.info("Loading trained model and generating Unity data...")
    
    # Check if spectral data exists
    if not Path(spectral_data_path).exists():
        logger.error(f"Spectral data directory not found: {spectral_data_path}")
        return
    
    # Check if trained model exists
    model_path = Path(neural_output_path) / "best_model.pth"
    if not model_path.exists():
        logger.error(f"Trained model not found: {model_path}")
        logger.error("Please run the training script first.")
        return
    
    # Create dataset
    dataset = SpectralDataset(spectral_data_path, max_files=100)  # Limit for testing
    
    if len(dataset) == 0:
        logger.error("No spectral data files found!")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=settings.processing.batch_size,
        shuffle=False,  # Keep order for consistent results
        num_workers=settings.processing.num_workers,
        drop_last=False,  # Don't drop last batch
        collate_fn=custom_collate_fn
    )
    
    # Load trained model
    logger.info(f"Loading model from {model_path}")
    model = load_trained_model(str(model_path), dataset.input_dim)
    
    # Move model to appropriate device
    device = 'auto'
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # Generate Unity data
    unity_data = generate_unity_data_from_model(model, dataloader)
    
    logger.info("Unity data generation complete!")
    logger.info(f"Generated {len(unity_data)} visual parameter sets")

if __name__ == "__main__":
    main() 