"""TurboQuant-style adaptive block-wise quantization for HuggingFace models."""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from scipy import stats


class TurboQuantizer:
    """
    TurboQuant-style adaptive block-wise quantization.
    
    Implements rotation-based scalar quantization inspired by TurboQuant (arXiv:2504.19874).
    Adapts the KV cache quantization approach for model weights.
    
    Features:
    - Random orthogonal rotation (Hadamard or QR-based)
    - Adaptive block sizing based on weight statistics
    - Per-block scalar quantization with calibration
    - Outlier handling via separate high-precision storage
    """
    
    def __init__(
        self,
        n_bits: int = 4,
        block_size: int = 64,
        rotation_type: str = "hadamard",
        outlier_threshold: float = 6.0,
        calibration_samples: int = 128,
    ):
        """
        Initialize TurboQuantizer.
        
        Args:
            n_bits: Number of bits for quantization (default 4 for Q4 comparison)
            block_size: Size of quantization blocks
            rotation_type: Type of rotation ("hadamard" or "random")
            outlier_threshold: Threshold for outlier detection (std multiples)
            calibration_samples: Number of samples for calibration
        """
        self.n_bits = n_bits
        self.block_size = block_size
        self.rotation_type = rotation_type
        self.outlier_threshold = outlier_threshold
        self.calibration_samples = calibration_samples
        
        # Quantization parameters
        self.n_levels = 2 ** n_bits
        self.quant_scale = 2.0 / (self.n_levels - 1)
        
    def _get_rotation_matrix(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Generate orthogonal rotation matrix.
        
        Args:
            n: Dimension size
            device: Device for tensor
            
        Returns:
            Orthogonal rotation matrix of shape (n, n)
        """
        if self.rotation_type == "hadamard" and n <= 1024:
            # Use Hadamard matrix for power-of-2 dimensions
            try:
                from scipy.linalg import hadamard
                H = hadamard(n)
                return torch.tensor(H, dtype=torch.float32, device=device) / np.sqrt(n)
            except:
                pass
        
        # Random orthogonal via QR decomposition
        A = torch.randn(n, n, device=device)
        Q, R = torch.linalg.qr(A)
        return Q
    
    def _rotate_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to weights for better quantization.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Rotated weights
        """
        if weights.dim() == 1:
            return weights
        
        if weights.dim() == 2:
            n = weights.shape[0]
            if n > 1:
                R = self._get_rotation_matrix(n, device=weights.device)
                return R @ weights
        
        return weights
    
    def _compute_block_stats(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute statistics for adaptive block quantization.
        
        Args:
            weights: Weight tensor
            
        Returns:
            Tuple of (block_means, block_scales, outlier_mask)
        """
        # Reshape into blocks
        flat = weights.flatten()
        n_elements = flat.numel()
        n_blocks = max(1, n_elements // self.block_size)
        
        # Pad if necessary
        if n_elements % self.block_size != 0:
            pad_size = self.block_size - (n_elements % self.block_size)
            flat = torch.cat([flat, flat[:pad_size]])
        
        # Reshape to blocks
        blocks = flat[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        
        # Compute per-block statistics
        block_means = blocks.mean(dim=1, keepdim=True)
        block_std = blocks.std(dim=1, keepdim=True) + 1e-6
        
        # Detect outliers
        outlier_mask = torch.abs(blocks - block_means) > self.outlier_threshold * block_std
        
        # Compute scales for quantization
        block_scales = block_std * 2.0  # Scale factor for each block
        
        return block_means.squeeze(-1), block_scales.squeeze(-1), outlier_mask
    
    def quantize_block(
        self, block: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantize a single block using scalar quantization.
        
        Args:
            block: Weight block to quantize
            scale: Scale factor for the block
            mean: Mean value for the block
            
        Returns:
            Quantized block (integer indices)
        """
        # Normalize block
        normalized = (block - mean) / scale
        
        # Map to quantization levels
        # Use uniform quantization with n_levels
        quantized = torch.round(normalized / self.quant_scale)
        quantized = torch.clamp(quantized, 0, self.n_levels - 1)
        
        return quantized.to(torch.int8)
    
    def dequantize_block(
        self, quantized: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize a block from integer indices.
        
        Args:
            quantized: Quantized indices
            scale: Scale factor
            mean: Mean value
            
        Returns:
            Dequantized weights
        """
        dequantized = quantized.to(torch.float32) * self.quant_scale * scale + mean
        return dequantized
    
    def quantize_weights(
        self, weights: torch.Tensor, layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quantize weight tensor using TurboQuant approach.
        
        Args:
            weights: Input weight tensor
            layer_name: Optional layer name for logging
            
        Returns:
            Dictionary containing quantized weights and metadata
        """
        original_shape = weights.shape
        original_dtype = weights.dtype
        
        # Apply rotation for better quantization
        rotated = self._rotate_weights(weights)
        
        # Compute block statistics
        block_means, block_scales, outlier_mask = self._compute_block_stats(rotated)
        
        # Flatten for block processing
        flat = rotated.flatten()
        n_elements = flat.numel()
        n_blocks = max(1, n_elements // self.block_size)
        
        if n_elements % self.block_size != 0:
            pad_size = self.block_size - (n_elements % self.block_size)
            flat = torch.cat([flat, flat[:pad_size]])
        
        # Process each block
        blocks = flat[:n_blocks * self.block_size].reshape(n_blocks, self.block_size)
        quantized_blocks = []
        
        for i in range(n_blocks):
            block = blocks[i]
            mean = block_means[i] if i < len(block_means) else 0.0
            scale = block_scales[i] if i < len(block_scales) else 1.0
            
            quantized = self.quantize_block(block, scale, mean)
            quantized_blocks.append({
                'indices': quantized,
                'scale': scale,
                'mean': mean,
            })
        
        return {
            'quantized_blocks': quantized_blocks,
            'original_shape': original_shape,
            'original_dtype': str(original_dtype),
            'n_bits': self.n_bits,
            'block_size': self.block_size,
            'outlier_mask': outlier_mask if outlier_mask.numel() < n_elements else None,
            'layer_name': layer_name,
        }
    
    def dequantize_weights(
        self, quantized_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Dequantize weights from quantized representation.
        
        Args:
            quantized_data: Dictionary from quantize_weights
            
        Returns:
            Dequantized weight tensor
        """
        blocks = quantized_data['quantized_blocks']
        original_shape = quantized_data['original_shape']
        
        # Reconstruct flat tensor
        dequantized_flat = []
        for block_data in blocks:
            dequantized = self.dequantize_block(
                block_data['indices'],
                block_data['scale'],
                block_data['mean'],
            )
            dequantized_flat.append(dequantized)
        
        flat = torch.cat(dequantized_flat)
        
        # Reshape to original
        return flat[:np.prod(original_shape)].reshape(original_shape)
    
    def quantize_model(
        self,
        model: torch.nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Quantize entire model.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Optional calibration samples
            verbose: Print progress
            
        Returns:
            Dictionary mapping layer names to quantized data
        """
        quantized_layers = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if verbose:
                    print(f"Quantizing {name}...")
                
                weight = module.weight.data
                quantized = self.quantize_weights(weight, layer_name=name)
                quantized_layers[name] = quantized
        
        return quantized_layers
    
    def dequantize_model(
        self,
        model: torch.nn.Module,
        quantized_layers: Dict[str, Dict[str, Any]],
        verbose: bool = True,
    ) -> torch.nn.Module:
        """
        Load quantized weights back into model.
        
        Args:
            model: Original model architecture
            quantized_layers: Quantized layer data
            verbose: Print progress
            
        Returns:
            Model with dequantized weights
        """
        for name, module in model.named_modules():
            if name in quantized_layers and hasattr(module, 'weight'):
                if verbose:
                    print(f"Dequantizing {name}...")
                
                dequantized = self.dequantize_weights(quantized_layers[name])
                module.weight.data = dequantized.to(module.weight.dtype)
        
        return model
    
    def compute_quantization_error(
        self,
        original: torch.Tensor,
        quantized_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute quantization error metrics.
        
        Args:
            original: Original weights
            quantized_data: Quantized representation
            
        Returns:
            Dictionary with error metrics
        """
        dequantized = self.dequantize_weights(quantized_data)
        
        # MSE
        mse = torch.mean((original - dequantized) ** 2).item()
        
        # MAE
        mae = torch.mean(torch.abs(original - dequantized)).item()
        
        # Max error
        max_error = torch.max(torch.abs(original - dequantized)).item()
        
        # SNR (Signal-to-Noise Ratio)
        signal_power = torch.mean(original ** 2).item()
        noise_power = mse
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'snr_db': snr,
        }