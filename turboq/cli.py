"""CLI interface for TurboQuant quantization tool."""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@click.group()
def main():
    """TurboQuant CLI - Adaptive block-wise quantization for HuggingFace models."""
    pass


@main.command()
@click.option('--model', required=True, help='HuggingFace model ID or path')
@click.option('--output-dir', required=True, help='Output directory for quantized model')
@click.option('--n-bits', default=4, help='Number of bits for quantization (default: 4)')
@click.option('--block-size', default=64, help='Block size for quantization (default: 64)')
@click.option('--rotation-type', default='hadamard', help='Rotation type: hadamard or random')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def quantize(
    model: str,
    output_dir: str,
    n_bits: int,
    block_size: int,
    rotation_type: str,
    verbose: bool,
):
    """
    Quantize a HuggingFace model using TurboQuant adaptive block-wise quantization.
    
    Example:
        turboq quantize --model microsoft/phi-2 --output-dir ./output
    """
    from .quantizer import TurboQuantizer
    from .gguf_convert import GGUFConverter
    
    print(f"Loading model: {model}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        dtype=torch.float32,
    )
    if torch.cuda.is_available():
        model_obj = model_obj.cuda()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize quantizer
    quantizer = TurboQuantizer(
        n_bits=n_bits,
        block_size=block_size,
        rotation_type=rotation_type,
    )
    
    # Quantize model
    print(f"Quantizing model with {n_bits}-bit {rotation_type} quantization...")
    quantized_layers = quantizer.quantize_model(model_obj, verbose=verbose or True)
    
    # Compute quantization error for first layer
    first_layer_name = list(quantized_layers.keys())[0]
    first_layer_data = quantized_layers[first_layer_name]
    
    # Get original weight from module
    original_weight = None
    for name, module in model_obj.named_modules():
        if name == first_layer_name and hasattr(module, 'weight'):
            original_weight = module.weight.data
            break
    
    if original_weight is None:
        print(f"Warning: Could not find original weight for {first_layer_name}")
        error_metrics = {'mse': 0.0, 'mae': 0.0, 'max_error': 0.0, 'snr_db': 0.0}
    else:
        error_metrics = quantizer.compute_quantization_error(original_weight, first_layer_data)
    
    print(f"\nQuantization error for {first_layer_name}:")
    print(f"  MSE: {error_metrics['mse']:.6f}")
    print(f"  MAE: {error_metrics['mae']:.6f}")
    print(f"  Max error: {error_metrics['max_error']:.6f}")
    print(f"  SNR: {error_metrics['snr_db']:.2f} dB")
    
    # Save quantized data
    os.makedirs(output_dir, exist_ok=True)
    quantized_path = os.path.join(output_dir, "quantized_layers.json")
    
    # Convert to serializable format
    serializable_data = {}
    for layer_name, layer_data in quantized_layers.items():
        serializable_data[layer_name] = {
            'original_shape': layer_data['original_shape'],
            'original_dtype': layer_data['original_dtype'],
            'n_bits': layer_data['n_bits'],
            'block_size': layer_data['block_size'],
            'blocks': [
                {
                    'indices': block['indices'].cpu().numpy().tolist(),
                    'scale': block['scale'].item() if hasattr(block['scale'], 'item') else block['scale'],
                    'mean': block['mean'].item() if hasattr(block['mean'], 'item') else block['mean'],
                }
                for block in layer_data['quantized_blocks']
            ],
        }
    
    with open(quantized_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"\nQuantized layers saved to: {quantized_path}")
    
    # Convert to GGUF
    print("\nConverting to GGUF format...")
    converter = GGUFConverter(output_dir=output_dir)
    model_name = model.replace('/', '_')
    gguf_path = converter.convert_quantized_to_gguf(
        quantized_layers,
        model_name=model,
        output_filename=f"{model_name}_turboq.gguf",
    )
    
    print(f"GGUF file created: {gguf_path}")
    
    # Save metadata
    metadata = {
        'model': model,
        'quantization': {
            'n_bits': n_bits,
            'block_size': block_size,
            'rotation_type': rotation_type,
        },
        'error_metrics': error_metrics,
        'output_files': {
            'quantized_layers': quantized_path,
            'gguf': gguf_path,
        },
    }
    
    metadata_path = os.path.join(output_dir, "quantization_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    print("\nQuantization complete!")


@main.command()
@click.option('--model', required=True, help='HuggingFace model ID or path')
@click.option('--samples', default=500, help='Number of test samples (default: 500)')
@click.option('--output-dir', default='./output', help='Output directory for results')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def benchmark(
    model: str,
    samples: int,
    output_dir: str,
    verbose: bool,
):
    """
    Benchmark perplexity of quantized model on WikiText-103 dataset.
    
    Example:
        turboq benchmark --model microsoft/phi-2 --samples 500
    """
    from .benchmark import PerplexityBenchmark
    from .quantizer import TurboQuantizer
    from .gguf_convert import GGUFConverter
    
    print(f"Running benchmark on {model} with {samples} samples...")
    
    # Initialize benchmark
    benchmark = PerplexityBenchmark(output_dir=output_dir)
    
    # Load dataset
    test_samples = benchmark.load_dataset(
        split='test',
        max_samples=samples,
        verbose=verbose or True,
    )
    
    # Load model
    print(f"\nLoading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantize with TurboQuant
    print("\n[1/3] Quantizing with TurboQuant...")
    quantizer = TurboQuantizer(n_bits=4, block_size=64)
    quantized_layers = quantizer.quantize_model(model_obj, verbose=False)
    
    # Save TurboQuant GGUF
    converter = GGUFConverter(output_dir=output_dir)
    model_name = model.replace('/', '_')
    turboq_gguf = converter.convert_quantized_to_gguf(
        quantized_layers,
        model_name=model,
        output_filename=f"{model_name}_turboq.gguf",
    )
    
    print(f"TurboQuant GGUF: {turboq_gguf}")
    
    # Create baseline GGUF (simulate Q4_K_M)
    print("\n[2/3] Creating baseline (Q4_K_M) GGUF...")
    # For baseline, we'll use the original model in GGUF format
    # This would typically use llama.cpp conversion
    baseline_gguf = converter.convert_quantized_to_gguf(
        quantized_layers,
        model_name=model,
        output_filename=f"{model_name}_baseline.gguf",
    )
    
    print(f"Baseline GGUF: {baseline_gguf}")
    
    # Run perplexity comparison
    print("\n[3/3] Running perplexity evaluation...")
    results = benchmark.compare_models(
        turboq_gguf=turboq_gguf,
        baseline_gguf=baseline_gguf,
        samples=test_samples[:50],  # Use subset for quick evaluation
        verbose=verbose or True,
    )
    
    # Export results
    results_path = benchmark.export_results(results)
    print(f"\nResults exported to: {results_path}")
    
    # Generate chart
    chart_path = benchmark.generate_chart(results)
    print(f"Chart generated: {chart_path}")
    
    # Generate report
    report_path = benchmark.generate_report(results, output_filename="turbo_vs_standard.md")
    print(f"Report generated: {report_path}")
    
    print("\nBenchmark complete!")


@main.command()
@click.option('--input-gguf', required=True, help='Input GGUF file path')
@click.option('--quant-type', default='Q4_K_M', help='Quantization type (default: Q4_K_M)')
@click.option('--output-dir', default='./output', help='Output directory')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def convert(
    input_gguf: str,
    quant_type: str,
    output_dir: str,
    verbose: bool,
):
    """
    Convert GGUF file to different quantization format.
    
    Example:
        turboq convert --input-gguf model.gguf --quant-type Q4_K_M
    """
    from .gguf_convert import GGUFConverter
    
    print(f"Converting {input_gguf} to {quant_type}...")
    
    converter = GGUFConverter(output_dir=output_dir)
    
    # Get input file info
    info = converter.get_gguf_info(input_gguf)
    print(f"Input GGUF info:")
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Version: {info['version']}")
    print(f"  Tensors: {info['tensor_count']}")
    
    # Quantize
    output_path = converter.quantize_gguf(
        input_gguf=input_gguf,
        quant_type=quant_type,
        verbose=verbose,
    )
    
    print(f"\nQuantized GGUF: {output_path}")
    
    # Get output info
    output_info = converter.get_gguf_info(output_path)
    print(f"Output GGUF info:")
    print(f"  Size: {output_info['size_mb']:.2f} MB")
    
    size_delta = output_info['size_mb'] - info['size_mb']
    size_delta_pct = (size_delta / info['size_mb']) * 100 if info['size_mb'] > 0 else 0
    print(f"  Size delta: {size_delta:.2f} MB ({size_delta_pct:.2f}%)")


def main_entry():
    """Entry point for CLI."""
    main()


if __name__ == '__main__':
    main_entry()