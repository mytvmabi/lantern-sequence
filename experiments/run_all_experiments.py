"""
Run All Experiments - Complete Artifact Evaluation

Master script to run all experiments from the paper.
Reproduces all tables and figures with one command.

Experiments:
1. LeNet-5 MNIST (50 rounds) - Table 2, ~20 min
2. ResNet-18 CIFAR-10 (30 rounds) - Table 2, ~2-3 hours with GPU
3. Detection Rate (all attack types) - Figure 4 & Table 3, ~45 min
4. Scalability (5-100 clients) - Figure 5, ~30 min

Total runtime: ~4-5 hours with GPU, ~15-20 hours without GPU

Usage:
  python run_all_experiments.py              # Run all experiments
  python run_all_experiments.py --quick      # Quick mode (fewer rounds)
  python run_all_experiments.py --skip-resnet # Skip ResNet (saves time)
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


def print_header(text):
    """Print formatted header"""
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)
    print()


def print_section(text):
    """Print formatted section"""
    print()
    print("-" * 70)
    print(text)
    print("-" * 70)
    print()


def run_experiment(script_name, args_list, description):
    """Run a single experiment script"""
    print_section(f"Running: {description}")
    
    cmd = [sys.executable, script_name] + args_list
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print()
        print(f"✓ Completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print()
        print(f"✗ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False, elapsed


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
        else:
            return False, "No GPU detected"
    except ImportError:
        return False, "PyTorch not installed"


def main():
    parser = argparse.ArgumentParser(description='Run All Experiments')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: fewer rounds for faster testing')
    parser.add_argument('--skip-resnet', action='store_true',
                       help='Skip ResNet-18 experiment (saves 2-3 hours)')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip detection rate experiment')
    parser.add_argument('--skip-scalability', action='store_true',
                       help='Skip scalability experiment')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for all results')
    args = parser.parse_args()
    
    print_header("RIV Protocol - Complete Artifact Evaluation")
    
    print("Configuration:")
    print(f"  Quick mode: {args.quick}")
    print(f"  Skip ResNet-18: {args.skip_resnet}")
    print(f"  Skip detection: {args.skip_detection}")
    print(f"  Skip scalability: {args.skip_scalability}")
    print(f"  Output directory: {args.output}")
    print()
    
    # Check GPU
    has_gpu, gpu_info = check_gpu()
    print(f"GPU Status: {gpu_info}")
    
    if not has_gpu and not args.skip_resnet:
        print()
        print("WARNING: No GPU detected. ResNet-18 will take 10-15 hours.")
        print("         Consider running with --skip-resnet or use GPU.")
        print()
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted. Run with --skip-resnet to skip ResNet experiment.")
            return
    
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    experiment_results = {
        'start_time': datetime.now().isoformat(),
        'config': vars(args),
        'gpu_info': gpu_info,
        'experiments': []
    }
    
    total_start = time.time()
    
    # Experiment 1: LeNet-5 MNIST
    print_header("Experiment 1: LeNet-5 on MNIST")
    print("Reproduces Table 2 results for LeNet-5")
    print("Expected accuracy: ~98%")
    print()
    
    lenet_args = ['--output', args.output]
    if args.quick:
        lenet_args.extend(['--rounds', '10'])
    
    success, elapsed = run_experiment(
        'lenet5_mnist.py',
        lenet_args,
        'LeNet-5 MNIST Training'
    )
    
    experiment_results['experiments'].append({
        'name': 'lenet5_mnist',
        'success': success,
        'time_minutes': elapsed / 60
    })
    
    # Experiment 2: ResNet-18 CIFAR-10
    if not args.skip_resnet:
        print_header("Experiment 2: ResNet-18 on CIFAR-10")
        print("Reproduces Table 2 results for ResNet-18")
        print("Expected accuracy: ~85-87%")
        print()
        
        resnet_args = ['--output', args.output]
        if args.quick:
            resnet_args.extend(['--rounds', '10'])
        
        success, elapsed = run_experiment(
            'resnet18_cifar10.py',
            resnet_args,
            'ResNet-18 CIFAR-10 Training'
        )
        
        experiment_results['experiments'].append({
            'name': 'resnet18_cifar10',
            'success': success,
            'time_minutes': elapsed / 60
        })
    
    # Experiment 3: Detection Rate
    if not args.skip_detection:
        print_header("Experiment 3: Free-Rider Detection Rate")
        print("Reproduces Figure 4 and Table 3")
        print("Tests detection of various attack types")
        print()
        
        detection_args = ['--output', args.output]
        if args.quick:
            detection_args.extend(['--rounds', '10', '--repetitions', '3'])
        
        success, elapsed = run_experiment(
            'detection_rate.py',
            detection_args,
            'Detection Rate Analysis'
        )
        
        experiment_results['experiments'].append({
            'name': 'detection_rate',
            'success': success,
            'time_minutes': elapsed / 60
        })
    
    # Experiment 4: Scalability
    if not args.skip_scalability:
        print_header("Experiment 4: Scalability Analysis")
        print("Reproduces Figure 5")
        print("Tests performance with increasing clients")
        print()
        
        scalability_args = ['--output', args.output]
        if args.quick:
            scalability_args.extend(['--client-counts', '5', '10', '20'])
        
        success, elapsed = run_experiment(
            'scalability.py',
            scalability_args,
            'Scalability Analysis'
        )
        
        experiment_results['experiments'].append({
            'name': 'scalability',
            'success': success,
            'time_minutes': elapsed / 60
        })
    
    total_time = time.time() - total_start
    experiment_results['end_time'] = datetime.now().isoformat()
    experiment_results['total_time_hours'] = total_time / 3600
    
    # Final summary
    print_header("Artifact Evaluation Complete")
    
    print("Experiment Summary:")
    print()
    
    for exp in experiment_results['experiments']:
        status = "✓" if exp['success'] else "✗"
        print(f"  {status} {exp['name']:<20} {exp['time_minutes']:>6.1f} minutes")
    
    print()
    print(f"Total time: {total_time/3600:.2f} hours")
    print()
    
    # Check for failures
    failed = [exp for exp in experiment_results['experiments'] if not exp['success']]
    if failed:
        print("⚠ Some experiments failed:")
        for exp in failed:
            print(f"  - {exp['name']}")
        print()
    else:
        print("✓ All experiments completed successfully!")
        print()
    
    # Save summary
    summary_file = output_dir / f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    print()
    
    # Find all result files
    print("Generated Result Files:")
    result_files = sorted(output_dir.glob('*.json'))
    for f in result_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print()
    print("Next Steps:")
    print("  1. Review result files in:", output_dir)
    print("  2. Compare with paper Table 2 (accuracy metrics)")
    print("  3. Compare with paper Figure 4 (detection rates)")
    print("  4. Compare with paper Figure 5 (scalability)")
    print()
    print("For questions or issues:")
    print("  - Check README.md for troubleshooting")
    print("  - Review individual experiment logs")
    print("  - Ensure all dependencies are installed")
    print()


if __name__ == '__main__':
    # Change to experiments directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    main()
