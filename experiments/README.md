# RIV Protocol Experiments

This directory contains all experiments from the paper for artifact evaluation.

## Quick Reference

| Experiment | Script | Runtime | Key Result |
|------------|--------|---------|------------|
| Demo | `lenet5_mnist_demo.py` | 2 min | Quick verification |
| LeNet-5 | `lenet5_mnist.py` | 20 min | 98% accuracy |
| ResNet-18 | `resnet18_cifar10.py` | 2-3 hr | 85-87% accuracy |
| Detection | `detection_rate.py` | 45 min | 95-99% detection |
| Scalability | `scalability.py` | 30 min | Linear scaling |

**Zero-Knowledge Mode**: Add `--zk` flag to any main experiment (lenet5, resnet18) to enable ZK proofs.

## Overview

All experiments are self-contained and can be run independently. Each script includes:
- Complete model implementation
- Data loading and partitioning
- RIV protocol integration
- Results collection and analysis
- Professional output formatting

## Verification Modes

The RIV protocol supports two modes:

### Transparent Mode (Default)
- Hash-based commitments (SHA-256)
- Fast verification: ~1-3ms per round
- Suitable for honest-but-curious server
- **Usage**: Run experiments normally (no flags needed)

### Zero-Knowledge Mode (Optional)
- KZG polynomial commitments on BLS12-381
- Full zero-knowledge proofs
- Slower: ~10-50ms per round
- Suitable for malicious server setting
- **Usage**: Add `--zk` flag to experiments

```bash
# Example: Enable ZK mode
python lenet5_mnist.py --zk
python resnet18_cifar10.py --zk
```

**Note**: Zero-knowledge mode requires compiling the Rust cryptographic backend. See main [README.md](../README.md) and [VERIFICATION_GUIDE.md](../VERIFICATION_GUIDE.md).

## Experiments

### 1. LeNet-5 on MNIST (`lenet5_mnist.py`)

Reproduces Table 2 results for LeNet-5.

```bash
# Full experiment (50 rounds, ~20 minutes)
python lenet5_mnist.py

# Quick test (10 rounds, ~5 minutes)
python lenet5_mnist.py --rounds 10

# With zero-knowledge mode
python lenet5_mnist.py --zk
```

**Expected Results:**
- Test accuracy: ~98%
- Commitment overhead: <5ms per round
- Verification time: <20ms per round
- Communication: ~5-6 KB per round

### 2. ResNet-18 on CIFAR-10 (`resnet18_cifar10.py`)

Reproduces Table 2 results for ResNet-18.

```bash
# Full experiment (30 rounds, ~2-3 hours with GPU)
python resnet18_cifar10.py

# Quick test (10 rounds, ~1 hour with GPU)
python resnet18_cifar10.py --rounds 10

# CPU warning: Will take 10-15 hours without GPU
```

**Expected Results:**
- Test accuracy: ~85-87%
- Commitment overhead: <10ms per round
- Verification time: <50ms per round
- Communication: ~15-20 KB per round

### 3. Detection Rate (`detection_rate.py`)

Reproduces Figure 4 and Table 3 - free-rider detection rates.

```bash
# Full experiment (all attacks, k ∈ {1,3,5,7,10}, ~45 minutes)
python detection_rate.py

# Quick test (fewer repetitions, ~20 minutes)
python detection_rate.py --rounds 10 --repetitions 3

# Specific attack types
python detection_rate.py --attack-types zero random

# Specific challenge budgets
python detection_rate.py --challenge-budgets 3 5 7
```

**Attack Types:**
- `zero`: Zero-gradient attack (no training)
- `random`: Random weight updates
- `stale`: Stale model from previous round
- `noise`: Gaussian noise injection

**Expected Results:**
- Zero-gradient: ~99% detection at k=5
- Random: ~95% detection at k=5
- Stale: ~90% detection at k=5
- Noise: ~70-85% detection at k=5

### 4. Scalability (`scalability.py`)

Reproduces Figure 5 - performance with increasing clients.

```bash
# Full experiment (5-100 clients, ~30 minutes)
python scalability.py

# Quick test (fewer client counts, ~15 minutes)
python scalability.py --client-counts 5 10 20

# Custom challenge budget
python scalability.py --challenge-budget 7
```

**Expected Results:**
- Commitment time: O(1) per client (~5-10ms)
- Proof generation: O(k) per client (~15-25ms)
- Verification: O(k) per client (~10-20ms)
- Near-linear scaling with client count

### 5. Demo (`lenet5_mnist_demo.py`)

Quick 5-round demo for initial testing.

```bash
# Run demo (~2 minutes)
python lenet5_mnist_demo.py
```

### 6. Run All Experiments (`run_all_experiments.py`)

Master script to run all experiments with one command.

```bash
# Run all experiments (~4-5 hours with GPU)
python run_all_experiments.py

# Quick mode (fewer rounds, ~1 hour)
python run_all_experiments.py --quick

# Skip ResNet (saves 2-3 hours)
python run_all_experiments.py --skip-resnet

# Skip specific experiments
python run_all_experiments.py --skip-detection --skip-scalability
```

## Command Line Options

All experiment scripts support common options:

- `--rounds N`: Number of training rounds (default varies by experiment)
- `--clients N`: Number of federated clients
- `--challenge-budget N`: Challenge budget k (default: 5)
- `--lr FLOAT`: Learning rate
- `--batch-size N`: Training batch size
- `--zk`: Enable zero-knowledge mode (slower, stronger privacy)
- `--output DIR`: Output directory for results (default: `results/`)

## Output Files

All experiments save results as JSON files with timestamps:

```
results/
├── lenet5_mnist_20250127_143022.json
├── resnet18_cifar10_20250127_150145.json
├── detection_rate_20250127_153330.json
├── scalability_20250127_160215.json
└── experiment_summary_20250127_163045.json
```

Each result file contains:
- Configuration parameters
- Per-round training metrics
- RIV protocol timing
- Model accuracy
- Detection/scalability statistics

## Viewing Results

```bash
# Pretty-print JSON results
cat results/lenet5_mnist_*.json | python -m json.tool

# Extract specific metrics
python -c "import json; data = json.load(open('results/lenet5_mnist_*.json')); print(data['rounds'][-1]['test_acc'])"

# Generate summary
python run_all_experiments.py  # Creates experiment_summary_*.json
```

## Reproducibility Notes

### Random Seeds
Experiments use fixed random seeds for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Data partitioning: Deterministic shuffling

### Platform Differences
Results may vary slightly across platforms:
- GPU vs CPU: Training speeds differ, final accuracy should match
- Different GPUs: CUDA optimizations may vary slightly
- OS differences: Minimal impact on results

### Expected Variance
- Accuracy: ±0.5% typical variance
- Timing: ±10% variance depending on system load
- Detection rates: ±2% variance due to randomness

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python lenet5_mnist.py --batch-size 32

# Reduce number of clients
python scalability.py --client-counts 5 10 20
```

### Slow Execution
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Use quick mode
python run_all_experiments.py --quick

# Skip expensive experiments
python run_all_experiments.py --skip-resnet
```

### Dataset Download Failures
```bash
# Manual download
cd ../data
bash download_datasets.sh

# Or specify different data directory
python lenet5_mnist.py --data-dir /path/to/data
```

### Import Errors
```bash
# Reinstall dependencies
cd ..
pip install -r requirements.txt

# Check installation
python -c "import torch; import numpy; print('OK')"
```

## Paper Cross-Reference

| Experiment | Paper Reference | Runtime | Key Metrics |
|------------|----------------|---------|-------------|
| `lenet5_mnist.py` | Table 2 (LeNet-5) | ~20 min | Accuracy: 98% |
| `resnet18_cifar10.py` | Table 2 (ResNet-18) | ~2-3 hr | Accuracy: 85-87% |
| `detection_rate.py` | Figure 4, Table 3 | ~45 min | Detection: 95-99% at k=5 |
| `scalability.py` | Figure 5 | ~30 min | Linear scaling |
| `lenet5_mnist_demo.py` | N/A (Demo) | ~2 min | Quick test |

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB
- Time: 15-20 hours total

### Recommended
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 4+ GB VRAM
- Storage: 10 GB
- Time: 4-5 hours total

## Quick Start

```bash
# 1. Run quick demo to verify setup
python lenet5_mnist_demo.py

# 2. Run one full experiment
python lenet5_mnist.py

# 3. Check results
ls -lh results/

# 4. Run all experiments
python run_all_experiments.py --quick
```

## Support

For issues or questions:
1. Check `../README.md` for general setup
2. Review experiment output for error messages
3. Check `results/` directory for partial results
4. Ensure all dependencies installed: `pip install -r ../requirements.txt`

## Citation

If you use these experiments, please cite our paper:

```bibtex
@inproceedings{riv2026,
  title={Retroactive Intermediate Value Verification for Federated Learning},
  author={Anonymous},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2026}
}
```
