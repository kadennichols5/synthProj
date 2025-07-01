"""
test script pre gen1 model training for simplicity
"""

import torch
import json
import numpy as np
from pathlib import Path
import logging

# Add config to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "config"))
from paths import get_spectral_data_path
from settings import settings

from neural_network import UnsupervisedVisualEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_outputs():
    """Test what the model outputs look like without training."""
    
    # Test with real spectral data first to get correct dimension
    spectral_data_path = get_spectral_data_path()
    actual_input_dim = None
    
    if Path(spectral_data_path).exists():
        # Find spectral files (excluding hidden files)
        spectral_files = [f for f in Path(spectral_data_path).glob("*.json") 
                         if not f.name.startswith('._')][:1]
        
        if spectral_files:
            try:
                with open(spectral_files[0], 'r') as f:
                    data = json.load(f)
                
                # Calculate actual input dimension from first byte
                for stem_name, stem_data in data['stems'].items():
                    if stem_data:
                        byte_data = stem_data[0]
                        magnitudes_len = len(byte_data['magnitudes'])
                        phases_len = len(byte_data['phases'])
                        actual_input_dim = 1 + 4 + magnitudes_len + phases_len  # time + stem_one_hot + magnitudes + phases
                        break
                
                logger.info(f"Calculated actual input dimension: {actual_input_dim}")
            except Exception as e:
                logger.warning(f"Could not read spectral file: {e}")
    
    # Use actual dimension or fallback
    input_dim = actual_input_dim or 4101
    logger.info(f"Using input dimension: {input_dim}")
    
    # Create a simple model
    model = UnsupervisedVisualEncoder(input_dim=input_dim)
    
    logger.info(f"Created model with input dimension: {input_dim}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create some dummy input data
    batch_size = 5
    dummy_input = torch.randn(batch_size, input_dim)
    
    logger.info(f"Input shape: {dummy_input.shape}")
    
    # Run the model
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Analyze outputs
    logger.info("\n=== MODEL OUTPUT ANALYSIS ===")
    
    for param_name, param_tensor in outputs.items():
        logger.info(f"\n{param_name.upper()}:")
        logger.info(f"  Shape: {param_tensor.shape}")
        logger.info(f"  Min: {param_tensor.min().item():.4f}")
        logger.info(f"  Max: {param_tensor.max().item():.4f}")
        logger.info(f"  Mean: {param_tensor.mean().item():.4f}")
        logger.info(f"  Std: {param_tensor.std().item():.4f}")
        
        # Show a few sample values
        if param_tensor.shape[0] > 0:
            sample_values = param_tensor[0].cpu().numpy()
            logger.info(f"  Sample values: {sample_values[:5].tolist()}")
    
    # Test the loss function
    logger.info("\n=== LOSS FUNCTION TEST ===")
    
    loss = model.unsupervised_loss(outputs, dummy_input)
    logger.info(f"Total loss: {loss.item():.4f}")
    
    # Test individual loss components
    constraints_loss = model.basic_constraints_loss(outputs)
    diversity_loss = model.simple_diversity_loss(outputs)
    
    logger.info(f"Constraints loss: {constraints_loss.item():.4f}")
    logger.info(f"Diversity loss: {diversity_loss.item():.4f}")
    
    # Test with real spectral data if available
    logger.info("\n=== REAL DATA TEST ===")
    
    if Path(spectral_data_path).exists():
        # Find spectral files (excluding hidden files)
        spectral_files = [f for f in Path(spectral_data_path).glob("*.json") 
                         if not f.name.startswith('._')][:3]
        
        if spectral_files:
            logger.info(f"Found {len(spectral_files)} spectral files")
            
            for i, file_path in enumerate(spectral_files):
                logger.info(f"\nTesting file {i+1}: {file_path.name}")
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract a few samples
                    sample_count = 0
                    for stem_name, stem_data in data['stems'].items():
                        if sample_count >= 2:  # Limit samples per file
                            break
                        
                        for byte_data in stem_data[:2]:  # Take first 2 bytes
                            # Create input vector
                            magnitudes = np.array(byte_data['magnitudes'])
                            phases = np.array(byte_data['phases'])
                            
                            # Normalize
                            magnitudes = magnitudes / (np.max(magnitudes) + 1e-8)
                            phases = phases / (np.max(np.abs(phases)) + 1e-8)
                            
                            # Create feature vector
                            time = byte_data['time']
                            stem_one_hot = [1.0, 0.0, 0.0, 0.0]  # Placeholder
                            
                            feature_vector = np.concatenate([[time], stem_one_hot, magnitudes, phases])
                            
                            # Ensure correct dimension
                            if len(feature_vector) == input_dim:
                                input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
                                
                                with torch.no_grad():
                                    real_outputs = model(input_tensor)
                                
                                logger.info(f"  {stem_name} at time {time:.2f}s:")
                                for param_name, param_tensor in real_outputs.items():
                                    values = param_tensor[0].cpu().numpy()
                                    logger.info(f"    {param_name}: {values[:3].tolist()}...")
                                
                                sample_count += 1
                            else:
                                logger.warning(f"    Dimension mismatch: {len(feature_vector)} != {input_dim}")
                                
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
        else:
            logger.info("No spectral files found")
    else:
        logger.info("Spectral data directory not found")
    
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Check if output ranges make sense for Unity")
    logger.info("2. Verify loss values are reasonable (not exploding)")
    logger.info("3. Ensure diversity loss is working (not all outputs the same)")
    logger.info("4. Consider what visual parameters you actually want")
    logger.info("5. Start training with this simple loss function")

if __name__ == "__main__":
    test_model_outputs() 