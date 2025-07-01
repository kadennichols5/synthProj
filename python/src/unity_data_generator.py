"""
Unity Data Generator
Generates real-time visual parameters for Unity visualization from spectral data.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Add config to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "config"))
from paths import get_spectral_data_path, get_project_info
from settings import settings

from neural_network import VisualEncoder, UnsupervisedVisualEncoder

logger = logging.getLogger(__name__)

class UnityDataGenerator:
    """Generates Unity-compatible visual data from spectral analysis."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 output_dir: str = None,
                 device: str = 'auto'):
        self.device = self._get_device(device)
        
        # Setup output directory
        if output_dir is None:
            output_dir = get_spectral_data_path().replace('spectral_data', 'unity_data')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model if provided
        self.model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No model provided, will use direct spectral mapping")
    
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
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get input dimension from checkpoint
        input_dim = checkpoint.get('input_dim', 4101)  # Default fallback
        
        # Create model with same architecture
        self.model = UnsupervisedVisualEncoder(input_dim=input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path} with input_dim={input_dim}")
    
    def spectral_to_visual_direct(self, spectral_data: Dict) -> Dict:
        """Convert spectral data directly to visual parameters without a trained model."""
        # Extract spectral features
        magnitudes = np.array(spectral_data['magnitudes'])
        phases = np.array(spectral_data['phases'])
        frequencies = np.array(spectral_data['frequencies'])
        
        # Normalize
        magnitudes = magnitudes / (np.max(magnitudes) + 1e-8)
        phases = phases / (np.max(np.abs(phases)) + 1e-8)
        
        # Calculate visual parameters based on spectral characteristics
        visual_params = {
            'shape': self._calculate_shape_from_spectrum(magnitudes, frequencies),
            'motion': self._calculate_motion_from_spectrum(magnitudes, phases),
            'texture': self._calculate_texture_from_spectrum(magnitudes, phases),
            'color': self._calculate_color_from_spectrum(magnitudes, frequencies),
            'brightness': self._calculate_brightness_from_spectrum(magnitudes),
            'position': self._calculate_position_from_spectrum(magnitudes, phases),
            'pattern': self._calculate_pattern_from_spectrum(magnitudes, frequencies)
        }
        
        return visual_params
    
    def _calculate_shape_from_spectrum(self, magnitudes: np.ndarray, frequencies: np.ndarray) -> List[float]:
        """Calculate shape parameters from spectral characteristics."""
        # Shape based on frequency distribution
        low_freq_energy = np.sum(magnitudes[frequencies < 1000])
        mid_freq_energy = np.sum(magnitudes[(frequencies >= 1000) & (frequencies < 5000)])
        high_freq_energy = np.sum(magnitudes[frequencies >= 5000])
        
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-8
        
        # Normalize and create shape vector
        shape = [
            low_freq_energy / total_energy,   # Rounded/soft shapes
            mid_freq_energy / total_energy,   # Medium complexity
            high_freq_energy / total_energy,  # Sharp/angular shapes
            0.1,  # Symmetry (placeholder)
            0.1,  # Regularity (placeholder)
            0.1   # Complexity (placeholder)
        ]
        
        return shape
    
    def _calculate_motion_from_spectrum(self, magnitudes: np.ndarray, phases: np.ndarray) -> List[float]:
        """Calculate motion parameters from spectral characteristics."""
        # Motion based on phase changes and magnitude variations
        phase_variance = np.var(phases)
        magnitude_variance = np.var(magnitudes)
        
        # Calculate velocity and acceleration proxies
        velocity = np.mean(np.abs(np.diff(magnitudes)))
        acceleration = np.mean(np.abs(np.diff(np.diff(magnitudes))))
        
        motion = [
            velocity,           # X velocity
            acceleration,       # X acceleration
            phase_variance,     # Y velocity
            magnitude_variance, # Y acceleration
            0.0,               # Z velocity
            0.0                # Z acceleration
        ]
        
        # Normalize to reasonable ranges
        motion = [np.clip(m, -1.0, 1.0) for m in motion]
        
        return motion
    
    def _calculate_texture_from_spectrum(self, magnitudes: np.ndarray, phases: np.ndarray) -> List[float]:
        """Calculate texture parameters from spectral characteristics."""
        # Texture based on spectral complexity
        spectral_entropy = -np.sum(magnitudes * np.log(magnitudes + 1e-8))
        phase_regularity = 1.0 / (1.0 + np.std(phases))
        
        texture = [
            spectral_entropy / 10.0,  # Roughness
            phase_regularity,         # Smoothness
            np.mean(magnitudes),      # Density
            np.std(magnitudes),       # Variation
            0.5,                      # Scale
            0.5,                      # Detail
            0.5,                      # Contrast
            0.5                       # Pattern
        ]
        
        # Normalize
        texture = [np.clip(t, 0.0, 1.0) for t in texture]
        
        return texture
    
        """Calculate brightness from spectral energy."""
        brightness = np.mean(magnitudes)
        return float(np.clip(brightness, 0.0, 1.0))
    
    def _calculate_position_from_spectrum(self, magnitudes: np.ndarray, phases: np.ndarray) -> List[float]:
        """Calculate position parameters from spectral characteristics."""
        # Position based on spectral center of mass and phase distribution
        spectral_center = np.average(np.arange(len(magnitudes)), weights=magnitudes)
        phase_center = np.average(phases)
        
        # Normalize to 3D space
        x = (spectral_center / len(magnitudes)) * 2 - 1  # -1 to 1
        y = (phase_center / np.pi) * 2 - 1  # -1 to 1
        z = np.mean(magnitudes) * 2 - 1  # -1 to 1
        
        position = [x, y, z]
        position = [np.clip(p, -1.0, 1.0) for p in position]
        
        return position
    
    def _calculate_pattern_from_spectrum(self, magnitudes: np.ndarray, frequencies: np.ndarray) -> List[float]:
        """Calculate pattern parameters from spectral characteristics."""
        # Pattern based on spectral peaks and regularity
        peaks = self._find_spectral_peaks(magnitudes)
        regularity = 1.0 / (1.0 + np.std(np.diff(peaks)))
        
        pattern = [
            len(peaks) / 10.0,  # Frequency
            regularity,         # Regularity
            np.max(magnitudes), # Intensity
            0.5,                # Symmetry
            0.5,                # Complexity
            0.5                 # Variation
        ]
        
        # Normalize
        pattern = [np.clip(p, 0.0, 1.0) for p in pattern]
        
        return pattern
    
    def _find_spectral_peaks(self, magnitudes: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in spectral magnitude."""
        peaks = []
        for i in range(1, len(magnitudes) - 1):
            if (magnitudes[i] > magnitudes[i-1] and 
                magnitudes[i] > magnitudes[i+1] and 
                magnitudes[i] > threshold * np.max(magnitudes)):
                peaks.append(i)
        return peaks
    
    def process_spectral_file(self, file_path: str) -> Dict:
        """Process a single spectral data file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        unity_data = {
            'segment_name': data.get('segment_name', Path(file_path).stem),
            'timestamp': datetime.now().isoformat(),
            'frames': []
        }
        
        # Process each stem
        for stem_name, stem_data in data['stems'].items():
            for byte_data in stem_data:
                if self.model:
                    # Use trained model
                    visual_params = self._model_inference(byte_data)
                else:
                    # Use direct spectral mapping
                    visual_params = self.spectral_to_visual_direct(byte_data)
                
                frame_data = {
                    'time': byte_data['time'],
                    'stem': stem_name,
                    'visual_params': visual_params
                }
                unity_data['frames'].append(frame_data)
        
        return unity_data
    
    def _model_inference(self, byte_data: Dict) -> Dict:
        """Run model inference on spectral data."""
        # Prepare input tensor
        magnitudes = np.array(byte_data['magnitudes'])
        phases = np.array(byte_data['phases'])
        
        # Normalize
        magnitudes = magnitudes / (np.max(magnitudes) + 1e-8)
        phases = phases / (np.max(np.abs(phases)) + 1e-8)
        
        # Create feature vector
        time = byte_data['time']
        stem_one_hot = [1.0, 0.0, 0.0, 0.0]  # Placeholder - should match training
        
        feature_vector = np.concatenate([[time], stem_one_hot, magnitudes, phases])
        
        # Verify dimension matches model expectation
        expected_dim = 1 + 4 + len(magnitudes) + len(phases)
        if self.model is not None:
            # Get the first layer's input dimension
            first_layer = list(self.model.encoder.children())[0]
            model_input_dim = first_layer.in_features
            if expected_dim != model_input_dim:
                logger.warning(f"Dimension mismatch: expected {expected_dim}, model expects {model_input_dim}")
        
        input_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Convert to Unity format
        visual_params = {
            'shape': outputs['shape'][0].cpu().numpy().tolist(),
            'motion': outputs['motion'][0].cpu().numpy().tolist(),
            'texture': outputs['texture'][0].cpu().numpy().tolist(),
            'color': outputs['color'][0].cpu().numpy().tolist(),
            'brightness': float(outputs['brightness'][0].cpu().numpy()),
            'position': outputs['position'][0].cpu().numpy().tolist(),
            'pattern': outputs['pattern'][0].cpu().numpy().tolist()
        }
        
        return visual_params
    
    def generate_unity_data_batch(self, 
                                 spectral_dir: str = None,
                                 output_file: str = 'unity_visual_data.json',
                                 max_files: int = None) -> str:
        """Generate Unity data from multiple spectral files."""
        if spectral_dir is None:
            spectral_dir = get_spectral_data_path()
        
        spectral_path = Path(spectral_dir)
        files = list(spectral_path.glob("*.json"))
        
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Processing {len(files)} spectral files...")
        
        all_unity_data = []
        
        for file_path in files:
            try:
                unity_data = self.process_spectral_file(str(file_path))
                all_unity_data.append(unity_data)
                logger.info(f"Processed {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Save combined data
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(all_unity_data, f, indent=2)
        
        logger.info(f"Unity data saved to {output_path}")
        logger.info(f"Generated data for {len(all_unity_data)} segments")
        
        return str(output_path)
    
    def generate_realtime_template(self, output_file: str = 'unity_realtime_template.cs') -> str:
        """Generate Unity C# template for real-time data loading."""
        template = '''using UnityEngine;
using System.Collections.Generic;
using System.IO;

[System.Serializable]
public class VisualParameters
{
    public float[] shape;
    public float[] motion;
    public float[] texture;
    public float[] color;
    public float brightness;
    public float[] position;
    public float[] pattern;
}

[System.Serializable]
public class VisualFrame
{
    public float time;
    public string stem;
    public VisualParameters visual_params;
}

[System.Serializable]
public class UnityVisualData
{
    public string segment_name;
    public string timestamp;
    public List<VisualFrame> frames;
}

public class AudioVisualizer : MonoBehaviour
{
    [Header("Data Loading")]
    public string dataFilePath = "unity_visual_data.json";
    public bool loadOnStart = true;
    
    [Header("Visualization")]
    public GameObject visualObject;
    public float updateRate = 60f;
    
    private List<UnityVisualData> visualData;
    private int currentSegment = 0;
    private int currentFrame = 0;
    private float lastUpdateTime = 0f;
    
    void Start()
    {
        if (loadOnStart)
        {
            LoadVisualData();
        }
    }
    
    void Update()
    {
        if (visualData != null && visualData.Count > 0)
        {
            UpdateVisualization();
        }
    }
    
    public void LoadVisualData()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, dataFilePath);
        
        if (File.Exists(filePath))
        {
            string jsonData = File.ReadAllText(filePath);
            visualData = JsonUtility.FromJson<List<UnityVisualData>>(jsonData);
            Debug.Log($"Loaded {visualData.Count} segments with visual data");
        }
        else
        {
            Debug.LogError($"Visual data file not found: {filePath}");
        }
    }
    
    void UpdateVisualization()
    {
        if (Time.time - lastUpdateTime < 1f / updateRate)
            return;
        
        lastUpdateTime = Time.time;
        
        if (currentSegment < visualData.Count && currentFrame < visualData[currentSegment].frames.Count)
        {
            VisualFrame frame = visualData[currentSegment].frames[currentFrame];
            ApplyVisualParameters(frame.visual_params);
            
            currentFrame++;
            
            // Move to next segment if needed
            if (currentFrame >= visualData[currentSegment].frames.Count)
            {
                currentSegment = (currentSegment + 1) % visualData.Count;
                currentFrame = 0;
            }
        }
    }
    
    void ApplyVisualParameters(VisualParameters params)
    {
        if (visualObject == null) return;
        
        // Apply shape
        if (params.shape != null && params.shape.Length >= 3)
        {
            // Modify object scale based on shape parameters
            Vector3 scale = new Vector3(params.shape[0], params.shape[1], params.shape[2]);
            visualObject.transform.localScale = scale;
        }
        
        // Apply position
        if (params.position != null && params.position.Length >= 3)
        {
            Vector3 position = new Vector3(params.position[0], params.position[1], params.position[2]);
            visualObject.transform.position = position;
        }
        
        // Apply color
        if (params.color != null && params.color.Length >= 4)
        {
            Color color = new Color(params.color[0], params.color[1], params.color[2], params.color[3]);
            Renderer renderer = visualObject.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = color;
            }
        }
        
        // Apply brightness
        if (renderer != null)
        {
            Color currentColor = renderer.material.color;
            currentColor *= params.brightness;
            renderer.material.color = currentColor;
        }
        
        // Apply motion
        if (params.motion != null && params.motion.Length >= 3)
        {
            Vector3 velocity = new Vector3(params.motion[0], params.motion[1], params.motion[2]);
            visualObject.transform.position += velocity * Time.deltaTime;
        }
    }
    
    public void SetVisualObject(GameObject obj)
    {
        visualObject = obj;
    }
    
    public void ResetVisualization()
    {
        currentSegment = 0;
        currentFrame = 0;
    }
}
'''
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(template)
        
        logger.info(f"Unity template saved to {output_path}")
        return str(output_path)

def main():
    """Main function for generating Unity data."""
    # Get project info
    project_info = get_project_info()
    logger.info("Project Configuration:")
    for key, value in project_info.items():
        logger.info(f"  {key}: {value}")
    
    # Create generator
    generator = UnityDataGenerator()
    
    # Generate Unity data
    output_path = generator.generate_unity_data_batch(max_files=10)  # Limit for testing
    
    # Generate Unity template
    template_path = generator.generate_realtime_template()
    
    logger.info(f"Unity data generation complete!")
    logger.info(f"Data file: {output_path}")
    logger.info(f"Template file: {template_path}")

if __name__ == "__main__":
    main() 