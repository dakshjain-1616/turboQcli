"""GGUF conversion utilities for TurboQuant quantized models."""

import os
import struct
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path


class GGUFConverter:
    """
    Convert quantized weights to GGUF format.
    
    GGUF (GGML Universal File) is the format used by llama.cpp for model storage.
    This converter creates GGUF files compatible with llama.cpp tools.
    """
    
    # GGUF magic number - "GGUF" as bytes in little-endian
    GGUF_MAGIC_BYTES = b'GGUF'
    
    # GGUF version
    GGUF_VERSION = 3
    
    # Tensor types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 4
    GGML_TYPE_Q5_1 = 5
    GGML_TYPE_Q8_0 = 6
    GGML_TYPE_Q8_1 = 7
    GGML_TYPE_Q4_K = 8
    GGML_TYPE_Q5_K = 9
    GGML_TYPE_Q6_K = 10
    GGML_TYPE_Q8_K = 11
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize GGUF converter.
        
        Args:
            output_dir: Directory for output GGUF files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _write_string(self, f, s: str):
        """Write a string to GGUF file."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_tensor_info(
        self,
        f,
        name: str,
        shape: List[int],
        tensor_type: int,
        offset: int,
    ):
        """Write tensor information to GGUF file."""
        self._write_string(f, name)
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<Q', dim))
        f.write(struct.pack('<I', tensor_type))
        f.write(struct.pack('<Q', offset))
    
    def convert_quantized_to_gguf(
        self,
        quantized_layers: Dict[str, Dict[str, Any]],
        model_name: str,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Convert quantized layers to GGUF format.
        
        Args:
            quantized_layers: Quantized layer data from TurboQuantizer
            model_name: Name of the model
            output_filename: Optional output filename
            
        Returns:
            Path to generated GGUF file
        """
        if output_filename is None:
            output_filename = f"{model_name.replace('/', '_')}_turboq.gguf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Collect tensor information
        tensors = []
        total_size = 0
        
        for layer_name, quant_data in quantized_layers.items():
            shape = quant_data['original_shape']
            if len(shape) == 2:
                # Weight matrix
                tensor_type = self.GGML_TYPE_Q4_K
                # Q4_K size: (n_rows * n_cols * 4 bits) / 8 bits per byte
                size_bytes = int(np.prod(shape) * 4 / 8)
                tensors.append({
                    'name': layer_name,
                    'shape': list(shape),
                    'type': tensor_type,
                    'size': size_bytes,
                    'data': quant_data,
                })
                total_size += size_bytes
        
        # Write GGUF file
        with open(output_path, 'wb') as f:
            # Write header - magic number as bytes (8 bytes: "GGUF" + 4 bytes version)
            f.write(self.GGUF_MAGIC_BYTES)  # 4 bytes: "GGUF"
            f.write(struct.pack('<I', self.GGUF_VERSION))  # 4 bytes: version
            
            # Write tensor count
            f.write(struct.pack('<Q', len(tensors)))
            
            # Write KV count (metadata)
            kv_count = 5  # General metadata keys
            f.write(struct.pack('<Q', kv_count))
            
            # Write metadata
            self._write_string(f, "general.name")
            f.write(struct.pack('<I', 1))  # Type: STRING
            self._write_string(f, model_name)
            
            self._write_string(f, "general.architecture")
            f.write(struct.pack('<I', 1))
            self._write_string(f, "transformer")
            
            self._write_string(f, "general.file_type")
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<I', 8))  # Q4_K
            
            self._write_string(f, "general.quantization_version")
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<I', 2))
            
            self._write_string(f, "general.size")
            f.write(struct.pack('<I', 1))
            self._write_string(f, str(total_size))
            
            # Write tensor info
            offset = 0
            for tensor in tensors:
                self._write_tensor_info(
                    f,
                    tensor['name'],
                    tensor['shape'],
                    tensor['type'],
                    offset,
                )
                offset += tensor['size']
            
            # Write tensor data (placeholder - actual quantized data)
            for tensor in tensors:
                # Write quantized block data
                quant_data = tensor['data']
                blocks = quant_data['quantized_blocks']
                
                # Serialize blocks
                for block_data in blocks:
                    indices = block_data['indices'].cpu().numpy()
                    f.write(indices.astype(np.uint8).tobytes())
        
        return output_path
    
    def convert_hf_to_gguf(
        self,
        model_path: str,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Convert HuggingFace model to GGUF format.
        
        This uses llama.cpp's conversion script if available.
        
        Args:
            model_path: Path to HuggingFace model directory
            output_filename: Optional output filename
            
        Returns:
            Path to generated GGUF file
        """
        # Check for llama.cpp conversion script
        convert_script = os.path.join(
            os.path.dirname(self.output_dir),
            "llama.cpp",
            "convert-hf-to-gguf.py",
        )
        
        if not os.path.exists(convert_script):
            raise FileNotFoundError(
                f"llama.cpp conversion script not found at {convert_script}. "
                "Please clone llama.cpp first."
            )
        
        if output_filename is None:
            model_name = os.path.basename(model_path)
            output_filename = f"{model_name}.gguf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Run conversion
        import subprocess
        result = subprocess.run(
            [
                "python",
                convert_script,
                model_path,
                "--output",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")
        
        return output_path
    
    def quantize_gguf(
        self,
        input_gguf: str,
        quant_type: str = "Q4_K_M",
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Quantize GGUF file using llama.cpp quantize tool.
        
        Args:
            input_gguf: Path to input GGUF file
            quant_type: Quantization type (e.g., "Q4_K_M")
            output_filename: Optional output filename
            
        Returns:
            Path to quantized GGUF file
        """
        # Check for llama.cpp quantize tool
        quantize_tool = os.path.join(
            os.path.dirname(self.output_dir),
            "llama.cpp",
            "build",
            "bin",
            "llama-quantize",
        )
        
        if not os.path.exists(quantize_tool):
            # Try alternative path
            quantize_tool = os.path.join(
                os.path.dirname(self.output_dir),
                "llama.cpp",
                "quantize",
            )
        
        if not os.path.exists(quantize_tool):
            raise FileNotFoundError(
                f"llama.cpp quantize tool not found at {quantize_tool}. "
                "Please build llama.cpp first."
            )
        
        if output_filename is None:
            base_name = os.path.basename(input_gguf).replace('.gguf', '')
            output_filename = f"{base_name}_{quant_type.lower()}.gguf"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Run quantization
        import subprocess
        result = subprocess.run(
            [
                quantize_tool,
                input_gguf,
                output_path,
                quant_type,
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Quantization failed: {result.stderr}")
        
        return output_path
    
    def get_gguf_info(self, gguf_path: str) -> Dict[str, Any]:
        """
        Get information about GGUF file.
        
        Args:
            gguf_path: Path to GGUF file
            
        Returns:
            Dictionary with GGUF metadata
        """
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        
        # Get file size
        file_size = os.path.getsize(gguf_path)
        
        # Read header
        with open(gguf_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
        
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic number: {magic}")
        
        return {
            'path': gguf_path,
            'size_bytes': file_size,
            'size_mb': file_size / (1024 * 1024),
            'version': version,
            'tensor_count': tensor_count,
            'kv_count': kv_count,
        }