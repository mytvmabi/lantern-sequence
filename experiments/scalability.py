"""
Scalability Experiment

Measure RIV protocol performance with increasing number of clients.
Reproduces Figure 5 results from the paper.

Metrics:
- Commitment time vs. number of clients
- Proof generation time vs. number of clients
- Verification time vs. number of clients
- Communication overhead vs. number of clients

Configuration:
- Model: LeNet-5 on MNIST
- Client counts: [5, 10, 20, 50, 100]
- Rounds: 5 per configuration (for timing averaging)
- Challenge budget: k=5
- Mode: Transparent (faster than ZK mode)

Expected Results:
- Commitment time: O(1) per client (~5-10ms)
- Proof generation: O(k) per client (~15-25ms)
- Verification: O(k) per client (~10-20ms)
- Communication: O(k) per client (~5-10 KB)
- Linear scaling with number of clients

Runtime: ~20-30 minutes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from riv_protocol import RIVProtocol


class LeNet5(nn.Module):
    """LeNet-5 architecture for MNIST"""
    
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_mnist_federated(num_clients, samples_per_client=1000, data_dir='../../data/mnist'):
    """Load and partition MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(train_dataset))
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, len(train_dataset))
        if start_idx < len(train_dataset):
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(train_dataset, client_indices))
    
    return client_datasets


def train_client(model, dataset, learning_rate=0.01, batch_size=64, device='cpu'):
    """Train client model for one round"""
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # No momentum for RIV verification
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    gradients = {}
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Save gradients from last batch
        if batch_idx == len(loader) - 1:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.detach().cpu().numpy().copy()
        
        optimizer.step()
    
    return gradients


def extract_parameters(model):
    """Extract model parameters as dictionary"""
    params = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            params[f'layer_{idx}'] = param.data.cpu().numpy().copy()
    return params


def estimate_communication(params):
    """Estimate communication size in bytes"""
    total_bytes = 0
    for key, value in params.items():
        total_bytes += value.nbytes
    return total_bytes


def run_scalability_experiment(num_clients, challenge_budget=5, num_rounds=5, device='cpu'):
    """Run scalability experiment for specific number of clients"""
    
    # Load data
    client_datasets = load_mnist_federated(
        num_clients=num_clients,
        samples_per_client=min(1000, 60000 // num_clients)
    )
    
    # Initialize model
    global_model = LeNet5().to(device)
    
    # Initialize RIV protocol
    riv = RIVProtocol(
        challenge_budget=challenge_budget,
        use_zero_knowledge=False,
        verbose=False
    )
    
    # Timing accumulators
    commit_times = []
    proof_times = []
    verify_times = []
    comm_sizes = []
    
    for round_idx in range(num_rounds):
        # Train all clients
        client_models = []
        client_gradients = []
        
        for dataset in client_datasets:
            client_model = LeNet5().to(device)
            client_model.load_state_dict(global_model.state_dict())
            gradients = train_client(client_model, dataset, device=device)
            client_models.append(client_model)
            client_gradients.append(gradients)
        
        # Extract parameters
        all_params = [extract_parameters(model) for model in client_models]
        
        # RIV protocol for each client
        round_commit_times = []
        round_proof_times = []
        round_verify_times = []
        round_comm_sizes = []
        
        for client_id, params in enumerate(all_params):
            metadata = {
                'round': round_idx,
                'client_id': client_id,
                'learning_rate': 0.01
            }
            
            # Commitment phase
            commit_result = riv.client_commit_phase(params, metadata)
            round_commit_times.append(commit_result['commit_time'])
            
            # Challenge phase
            challenge_result = riv.server_challenge_phase(
                commitment_hash=commit_result['commitment_hash'],
                num_layers=len(params),
                metadata=metadata
            )
            
            # Proof phase
            layer_data = {}
            gradients = client_gradients[client_id]
            
            for layer_idx in challenge_result['challenged_layers']:
                layer_name = f'layer_{layer_idx}'
                if layer_name in params:
                    # Map layer name to parameter name for gradient lookup
                    param_name = list(client_models[client_id].named_parameters())[layer_idx][0]
                    gradient = gradients.get(param_name, np.zeros_like(params[layer_name]))
                    
                    layer_data[layer_name] = {
                        'activation': params[layer_name],
                        'gradient': gradient,
                        'weight_old': params[layer_name],
                        'weight_new': params[layer_name]
                    }
            
            proof_result = riv.client_proof_phase(
                challenged_layers=challenge_result['challenged_layers'],
                layer_data=layer_data,
                training_config=metadata
            )
            round_proof_times.append(proof_result['proof_time'])
            
            # Verification phase
            verify_result = riv.server_verify_phase(
                commitments=commit_result['commitments'],
                proofs=proof_result['proofs'],
                challenged_layers=challenge_result['challenged_layers'],
                metadata=metadata
            )
            round_verify_times.append(verify_result['verification_time'])
            
            # Estimate communication
            comm_size = estimate_communication(layer_data)
            round_comm_sizes.append(comm_size)
        
        # Aggregate round statistics
        commit_times.append(np.mean(round_commit_times))
        proof_times.append(np.mean(round_proof_times))
        verify_times.append(np.mean(round_verify_times))
        comm_sizes.append(np.mean(round_comm_sizes))
    
    # Return average metrics
    return {
        'commit_time_ms': np.mean(commit_times) * 1000,
        'proof_time_ms': np.mean(proof_times) * 1000,
        'verify_time_ms': np.mean(verify_times) * 1000,
        'comm_size_kb': np.mean(comm_sizes) / 1024,
        'commit_std': np.std(commit_times) * 1000,
        'proof_std': np.std(proof_times) * 1000,
        'verify_std': np.std(verify_times) * 1000,
        'comm_std': np.std(comm_sizes) / 1024
    }


def main():
    parser = argparse.ArgumentParser(description='Scalability Experiment')
    parser.add_argument('--client-counts', nargs='+', type=int, default=[5, 10, 20, 50, 100],
                       help='Client counts to test')
    parser.add_argument('--challenge-budget', type=int, default=5, help='Challenge budget k')
    parser.add_argument('--rounds', type=int, default=5, help='Rounds per configuration')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Scalability Experiment")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Client counts: {args.client_counts}")
    print(f"  Challenge budget: {args.challenge_budget}")
    print(f"  Rounds per configuration: {args.rounds}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Results storage
    results = {
        'config': vars(args),
        'scalability': {}
    }
    
    print(f"Running {len(args.client_counts)} scalability tests...")
    print()
    
    start_time = time.time()
    
    # Run experiments for each client count
    for idx, num_clients in enumerate(args.client_counts):
        print(f"Testing {num_clients} clients ({idx+1}/{len(args.client_counts)})...")
        
        metrics = run_scalability_experiment(
            num_clients=num_clients,
            challenge_budget=args.challenge_budget,
            num_rounds=args.rounds,
            device=device
        )
        
        results['scalability'][f'clients_{num_clients}'] = metrics
        
        print(f"  Commitment: {metrics['commit_time_ms']:.2f} ± {metrics['commit_std']:.2f} ms")
        print(f"  Proof gen:  {metrics['proof_time_ms']:.2f} ± {metrics['proof_std']:.2f} ms")
        print(f"  Verification: {metrics['verify_time_ms']:.2f} ± {metrics['verify_std']:.2f} ms")
        print(f"  Communication: {metrics['comm_size_kb']:.2f} ± {metrics['comm_std']:.2f} KB")
        print()
    
    total_time = time.time() - start_time
    
    # Summary
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    
    print("Scalability Summary:")
    print()
    print(f"{'Clients':<10} {'Commit (ms)':<15} {'Proof (ms)':<15} {'Verify (ms)':<15} {'Comm (KB)':<15}")
    print("-" * 70)
    
    for num_clients in args.client_counts:
        key = f'clients_{num_clients}'
        m = results['scalability'][key]
        print(f"{num_clients:<10} {m['commit_time_ms']:<15.2f} {m['proof_time_ms']:<15.2f} " +
              f"{m['verify_time_ms']:<15.2f} {m['comm_size_kb']:<15.2f}")
    
    print()
    
    # Analyze scaling
    print("Scaling Analysis:")
    baseline = results['scalability'][f'clients_{args.client_counts[0]}']
    largest = results['scalability'][f'clients_{args.client_counts[-1]}']
    
    ratio = args.client_counts[-1] / args.client_counts[0]
    commit_scaling = largest['commit_time_ms'] / baseline['commit_time_ms']
    proof_scaling = largest['proof_time_ms'] / baseline['proof_time_ms']
    verify_scaling = largest['verify_time_ms'] / baseline['verify_time_ms']
    
    print(f"  Client count increased: {ratio:.1f}x")
    print(f"  Commitment time increased: {commit_scaling:.2f}x")
    print(f"  Proof time increased: {proof_scaling:.2f}x")
    print(f"  Verification time increased: {verify_scaling:.2f}x")
    
    if commit_scaling < ratio * 1.5 and proof_scaling < ratio * 1.5 and verify_scaling < ratio * 1.5:
        print("  ✓ Near-linear scaling confirmed")
    else:
        print("  ⚠ Scaling may be super-linear (check results)")
    
    print()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'scalability_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    print("Next steps:")
    print("  - View results: cat", results_file)
    print("  - Run detection experiments: python experiments/detection_rate.py")
    print("  - Run full training: python experiments/lenet5_mnist.py")


if __name__ == '__main__':
    main()
