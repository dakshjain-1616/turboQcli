# turboq-cli: TurboQuant Quantization CLI Tool

## Goal
Build a CLI tool `turboq` that implements TurboQuant-style adaptive block-wise weight quantization, quantizes HuggingFace models (≤7B), and benchmarks perplexity against Q4_K_M baseline on WikiText-103.

## Research Summary

**TurboQuant Algorithm** (arXiv:2504.19874, ICLR 2026):
- Core idea: Random orthogonal rotation → scalar quantization on Beta-distributed coordinates
- For KV cache: 3-bit keys, 2-bit values with QJL residual coding
- Achieves near-optimal distortion rates with data-oblivious algorithms
- Implementation exists at `0xSero/turboquant` (KV cache focused, Triton kernels)

**Weight Quantization Approach** (adapted for this task):
- Since TurboQuant paper focuses on KV cache, I'll implement adaptive block-wise quantization inspired by the rotation + scalar quantization principle
- Use calibration data (WikiText samples) to determine optimal quantization parameters per block
- Compare against standard Q4_K_M (llama.cpp GGUF format)

**WikiText-103 Dataset**:
- HuggingFace ID: `Salesforce/wikitext` (wikitext-103-v1 config)
- Standard perplexity evaluation benchmark
- Will use 500 samples for evaluation

**llama.cpp Perplexity**:
- Command: `./perplexity -m model.gguf -f test-corpus.txt`
- Need to build llama.cpp, convert models to GGUF, run perplexity evaluation

**Q4_K_M Format**:
- Standard llama.cpp 4-bit quantization with K-quants
- Good balance of quality/speed/size

## Approach

1. **TurboQuant Implementation**: Implement adaptive block-wise quantization with:
   - Random rotation (Hadamard or orthogonal)
   - Block-wise scalar quantization with calibration
   - 4-bit target (matching Q4_K_M comparison)
   - Save in GGUF-compatible format or custom format convertible to GGUF

2. **CLI Tool**: Build with `click` for:
   - `turboq quantize --model <hf_id> --output-dir <path>`
   - `turboq benchmark --model <hf_id> --samples 500`
   - Download models via `huggingface-hub`
   - Run quantization, save outputs

3. **GGUF Conversion**: 
   - Use `llama.cpp` tools to convert quantized weights to GGUF
   - Generate both turboq and Q4_K_M versions

4. **Perplexity Benchmark**:
   - Load WikiText-103 samples
   - Run llama.cpp perplexity on both GGUFs
   - Collect metrics: perplexity, size, tokens/s

5. **Reporting**:
   - Generate markdown report with delta metrics
   - Create matplotlib chart showing perplexity per 50-sample window

## Subtasks

1. **Setup project structure**
   - Create `turboq/` package directory
   - Create `pyproject.toml` with dependencies (click, transformers, datasets, matplotlib, huggingface-hub, numpy)
   - Create initial `README.md` skeleton

2. **Implement TurboQuant quantizer** (`turboq/quantizer.py`)
   - Adaptive block-wise quantization with rotation
   - Calibration using sample data
   - 4-bit quantization output
   - GGUF export functionality

3. **Build CLI interface** (`turboq/cli.py`)
   - `quantize` command with model ID, output dir options
   - `benchmark` command for running perplexity evaluation
   - Progress reporting, error handling

4. **Implement benchmark module** (`turboq/benchmark.py`)
   - Load WikiText-103 dataset
   - Interface with llama.cpp perplexity tool
   - Collect and aggregate metrics

5. **Build llama.cpp integration**
   - Clone and build llama.cpp
   - Create conversion scripts for GGUF output
   - Run Q4_K_M baseline quantization

6. **Run benchmark on sample model** (e.g., `microsoft/phi-2` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
   - Quantize with TurboQuant
   - Quantize with Q4_K_M
   - Run perplexity evaluation
   - Generate report and chart

7. **Generate final deliverables**
   - `turbo_vs_standard.md` report
   - `perplexity_chart.png`
   - Complete `README.md` with usage examples

## Deliverables

| File Path | Description |
|-----------|-------------|
| `/root/projects/turboq_cli/pyproject.toml` | Package config, installable via pip |
| `/root/projects/turboq_cli/turboq/__init__.py` | Package init |
| `/root/projects/turboq_cli/turboq/cli.py` | Click CLI interface |
| `/root/projects/turboq_cli/turboq/quantizer.py` | TurboQuant implementation |
| `/root/projects/turboq_cli/turboq/benchmark.py` | Perplexity benchmark logic |
| `/root/projects/turboq_cli/turboq/gguf_convert.py` | GGUF conversion utilities |
| `/root/projects/turboq_cli/README.md` | Documentation with install/usage |
| `/root/projects/turboq_cli/turbo_vs_standard.md` | Benchmark comparison report |
| `/root/projects/turboq_cli/perplexity_chart.png` | Perplexity visualization |

## Evaluation Criteria

- CLI tool installs via `pip install -e .` and runs `turboq --help`
- `turboq quantize --model <model_id> --output-dir <path>` produces GGUF files
- Benchmark runs on WikiText-103 with 500 samples
- Report shows perplexity delta, size delta, speed comparison
- Chart shows perplexity per 50-sample window for both quantizations
- Works on models ≤7B parameters

## Notes

- llama.cpp needs to be cloned and built in project directory
- Need ~10GB disk space for model downloads and GGUF files
- GPU not required for perplexity evaluation (CPU inference via llama.cpp)
- Will use a small model (≤2B) for initial benchmark to keep runtime reasonable
- TurboQuant weight quantization is adapted from KV cache approach - may need tuning for optimal results