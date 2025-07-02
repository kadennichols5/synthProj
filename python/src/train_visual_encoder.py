"""
Neural Network Training Pipeline for Visual Encoder
Trains the VisualEncoder on spectral data and saves results for Unity integration.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import shutil

# Add config to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "config"))
from paths import get_spectral_data_path, get_neural_output_path, get_project_info
from settings import settings

from neural_network import VisualEncoder, UnsupervisedVisualEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length spectral data.
    Pads sequences to the maximum length in the batch.
    """
    # Find the maximum sequence length in this batch
    max_length = max(tensor.shape[0] for tensor in batch)
    batch_size = len(batch)
    feature_dim = batch[0].shape[1]
    
    # Create padded tensor
    padded_batch = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    
    # Fill in the actual data
    for i, tensor in enumerate(batch):
        length = tensor.shape[0]
        padded_batch[i, :length, :] = tensor
    
    return padded_batch

def get_directory_size(directory_path: Path) -> Tuple[int, str]:
    """Get the total size of a directory in bytes and human-readable format."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.warning(f"Could not calculate directory size: {e}")
        return 0, "0 B"
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return total_size, f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return total_size * 1024**3, f"{total_size:.1f} TB"

def log_storage_usage(directory: Path, description: str = ""):
    """Log the storage usage of a directory."""
    if directory.exists():
        size_bytes, size_readable = get_directory_size(directory)
        logger.info(f"Storage usage {description}: {size_readable} ({size_bytes} bytes)")
        return size_bytes
    else:
        logger.info(f"Directory does not exist: {directory}")
        return 0

class SpectralDataset(Dataset):
    """Dataset for loading spectral data from JSON files."""
    
    def __init__(self, spectral_data_dir: str, max_files: Optional[int] = None):
        self.spectral_data_dir = Path(spectral_data_dir)
        # Filter out hidden files and ensure they're JSON files
        self.files = [f for f in self.spectral_data_dir.glob("*.json") 
                     if not f.name.startswith('.') and f.is_file()]
        
        if max_files:
            self.files = self.files[:max_files]
        
        logger.info(f"Found {len(self.files)} spectral data files")
        
        # Calculate input dimension from first file
        if self.files:
            self.input_dim = self._calculate_input_dim()
            logger.info(f"Input dimension: {self.input_dim}")
        else:
            self.input_dim = 4101  # Default: 1 + 4 + 2048 + 2048
    
    def _calculate_input_dim(self) -> int:
        """Calculate the input dimension from the first file."""
        try:
            with open(self.files[0], 'r') as f:
                data = json.load(f)
            
            # Get dimensions from first stem's first byte
            for stem_name, stem_data in data['stems'].items():
                if stem_data:
                    byte_data = stem_data[0]
                    magnitudes_len = len(byte_data['magnitudes'])
                    phases_len = len(byte_data['phases'])
                    
                    # Calculate: time(1) + stem_one_hot(4) + magnitudes + phases
                    input_dim = 1 + 4 + magnitudes_len + phases_len
                    logger.info(f"Calculated input dimension: {input_dim} (magnitudes: {magnitudes_len}, phases: {phases_len})")
                    return input_dim
            
            # Fallback
            return 4101
        except Exception as e:
            logger.warning(f"Could not calculate input dimension from file: {e}")
            return 4101
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading file {file_path.name}: {e}")
            # Return empty tensor with correct dimensions
            return torch.zeros((1, self.input_dim), dtype=torch.float32)
        
        # Extract spectral data from all stems
        spectral_vectors = []
        
        for stem_name, stem_data in data['stems'].items():
            for byte_data in stem_data:
                # Combine magnitudes and phases
                magnitudes = np.array(byte_data['magnitudes'])
                phases = np.array(byte_data['phases'])
                
                # Normalize
                magnitudes = magnitudes / (np.max(magnitudes) + 1e-8)
                phases = phases / (np.max(np.abs(phases)) + 1e-8)
                
                # Create feature vector: [time, stem_one_hot, magnitudes, phases]
                time = byte_data['time']
                stem_one_hot = self._get_stem_one_hot(stem_name)
                
                feature_vector = np.concatenate([
                    [time],
                    stem_one_hot,
                    magnitudes,
                    phases
                ])
                
                spectral_vectors.append(feature_vector)
        
        # Convert to tensor
        if spectral_vectors:
            # Convert list to numpy array first to avoid slow tensor creation warning
            spectral_array = np.array(spectral_vectors)
            return torch.tensor(spectral_array, dtype=torch.float32)
        else:
            # Return empty tensor with correct dimensions
            return torch.zeros((1, self.input_dim), dtype=torch.float32)
    
    def _get_stem_one_hot(self, stem_name: str) -> np.ndarray:
        """Convert stem name to one-hot encoding."""
        stems = ['vocals', 'drums', 'bass', 'other']
        one_hot = np.zeros(4)
        if stem_name in stems:
            one_hot[stems.index(stem_name)] = 1.0
        return one_hot

class VisualEncoderTrainer:
    """Trainer class for the VisualEncoder."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 output_dir: str = None):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # Setup output directory
        if output_dir is None:
            output_dir = get_neural_output_path()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        
        # Storage monitoring
        self.initial_storage = log_storage_usage(self.output_dir, "at initialization")
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the best available device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def setup_training(self, 
                      learning_rate: float = 1e-4,
                      weight_decay: float = 1e-5):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, spectral_data in enumerate(progress_bar):
            spectral_data = spectral_data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, UnsupervisedVisualEncoder):
                outputs = self.model(spectral_data)
                loss = self.model.unsupervised_loss(outputs, spectral_data)
            else:
                # For regular VisualEncoder, we need a different approach
                # This would require ground truth visual parameters
                logger.warning("Regular VisualEncoder requires ground truth targets")
                outputs = self.model(spectral_data)
                # Use a simple regularization loss as fallback
                loss = torch.mean(torch.stack([torch.mean(output) for output in outputs.values()]))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for spectral_data in dataloader:
                spectral_data = spectral_data.to(self.device)
                
                if isinstance(self.model, UnsupervisedVisualEncoder):
                    outputs = self.model(spectral_data)
                    loss = self.model.unsupervised_loss(outputs, spectral_data)
                else:
                    # Same fallback as training
                    outputs = self.model(spectral_data)
                    loss = torch.mean(torch.stack([torch.mean(output) for output in outputs.values()]))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader = None,
              num_epochs: int = 100,
              save_interval: int = 10):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model.pth')
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}"
            if val_loss:
                log_msg += f" - Val Loss: {val_loss:.4f}"
            logger.info(log_msg)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                self.save_training_history()
    
    def save_model(self, filename: str):
        """Save the model."""
        save_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'settings': settings.to_dict(),
            'input_dim': self.model.encoder[0].in_features  # Save input dimension
        }, save_path)
        
        # Monitor storage after saving
        current_storage = log_storage_usage(self.output_dir, f"after saving {filename}")
        if self.initial_storage > 0:
            increase = current_storage - self.initial_storage
            increase_readable = f"{increase / 1024 / 1024:.1f} MB" if increase > 1024*1024 else f"{increase / 1024:.1f} KB"
            logger.info(f"Storage increase since initialization: {increase_readable}")
        
        logger.info(f"Model saved to {save_path}")
    
    def save_training_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'settings': settings.to_dict()
        }
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate_unity_data(self, dataloader: DataLoader, output_file: str = 'unity_visual_data.json'):
        """Generate Unity-compatible visual data."""
        self.model.eval()
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
                    single_timestep = single_timestep.to(self.device)
                    outputs = self.model(single_timestep)
                    
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
        unity_path = self.output_dir / output_file
        with open(unity_path, 'w') as f:
            json.dump(unity_data, f, indent=2)
        
        # Monitor storage after generating Unity data
        final_storage = log_storage_usage(self.output_dir, "after generating Unity data")
        if self.initial_storage > 0:
            total_increase = final_storage - self.initial_storage
            total_increase_readable = f"{total_increase / 1024 / 1024:.1f} MB" if total_increase > 1024*1024 else f"{total_increase / 1024:.1f} KB"
            logger.info(f"Total storage increase: {total_increase_readable}")
        
        logger.info(f"Unity data saved to {unity_path}")
        logger.info(f"Generated {len(unity_data)} visual parameter sets")
        
        return unity_data

def main():
    """Main training function."""
    # Get paths
    spectral_data_path = get_spectral_data_path()
    project_info = get_project_info()
    
    logger.info("Project Configuration:")
    for key, value in project_info.items():
        logger.info(f"  {key}: {value}")
    
    # Monitor initial storage state
    neural_output_path = Path(get_neural_output_path())
    logger.info("=== STORAGE MONITORING START ===")
    log_storage_usage(neural_output_path, "before training")
    
    # Check if spectral data exists
    if not Path(spectral_data_path).exists():
        logger.error(f"Spectral data directory not found: {spectral_data_path}")
        logger.error("Please run the spectral analysis pipeline first.")
        return
    
    # Create dataset
    dataset = SpectralDataset(spectral_data_path, max_files=100)  # Limit for testing
    
    if len(dataset) == 0:
        logger.error("No spectral data files found!")
        return
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=settings.processing.batch_size,
        shuffle=True,
        num_workers=settings.processing.num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=settings.processing.batch_size,
        shuffle=False,
        num_workers=settings.processing.num_workers,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    # Create model with correct input dimension
    input_dim = dataset.input_dim
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = UnsupervisedVisualEncoder(input_dim=input_dim)
    
    # Create trainer
    trainer = VisualEncoderTrainer(model, device='auto')
    trainer.setup_training()
    
    # Train model
    trainer.train(train_dataloader, val_dataloader, num_epochs=50, save_interval=10)
    
    # Generate Unity data
    trainer.generate_unity_data(train_dataloader)
    
    # Final storage monitoring
    logger.info("=== STORAGE MONITORING SUMMARY ===")
    log_storage_usage(neural_output_path, "after complete training")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 