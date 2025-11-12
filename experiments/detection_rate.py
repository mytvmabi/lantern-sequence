"""
Free-Rider Detection Rate Experiment

Measure RIV protocol's ability to detect free-rider attacks.
Reproduces Figure 4 and Table 3 results from the paper.

Attack types:
1. Zero-gradient: Client reports zero gradients (no training)
2. Random: Client reports random weight updates
3. Stale: Client reports old model from previous round
4. Noise: Client adds Gaussian noise to genuine updates

Configuration:
- Model: LeNet-5 on MNIST
- Clients: 5 (1 malicious)
- Rounds: 20 per attack type
- Challenge budgets: k ∈ {1, 3, 5, 7, 10}
- Detection threshold: Statistical deviation > 3σ

Expected Results:
- Zero-gradient: ~99% detection at k=5
- Random: ~95% detection at k=5
- Stale: ~90% detection at k=5
- Noise: ~70-85% detection at k=5 (depends on noise level)

Runtime: ~30-45 minutes
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


def load_mnist_federated(num_clients=5, samples_per_client=10000, data_dir='../../data/mnist'):
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


def apply_attack(params, attack_type, honest_params=None, noise_std=0.1):
    """Apply free-rider attack to model parameters
    
    Args:
        params: Attacker's parameters (potentially fake)
        attack_type: Type of attack to simulate
        honest_params: Reference honest parameters for realistic attacks
        noise_std: Standard deviation for noise-based attacks
    """
    attacked = {}
    
    if attack_type == 'zero':
        # Zero-gradient attack: claim training but provide zeros
        # This simulates a client that didn't actually train
        for key, value in params.items():
            attacked[key] = np.zeros_like(value)
    
    elif attack_type == 'random':
        # Random model attack: provide random weights
        # Simulates fabricated model without real training
        for key, value in params.items():
            attacked[key] = np.random.randn(*value.shape) * 0.01
    
    elif attack_type == 'stale':
        # Stale model attack: return old parameters unchanged
        # Simulates reusing previous round's model
        attacked = params.copy()
    
    elif attack_type == 'noise':
        # Additive noise attack: corrupt honest parameters
        # Simulates Byzantine adversary with noise injection
        if honest_params is not None:
            for key in params.keys():
                noise = np.random.randn(*params[key].shape) * noise_std
                attacked[key] = honest_params.get(key, params[key]) + noise
        else:
            for key, value in params.items():
                noise = np.random.randn(*value.shape) * noise_std
                attacked[key] = value + noise
    
    elif attack_type == 'model_replacement':
        # Model replacement: use a different pretrained model
        # Simulates sophisticated attack with plausible but incorrect model
        if honest_params is not None:
            for key in params.keys():
                # Mix honest and random parameters
                attacked[key] = 0.7 * honest_params.get(key, params[key]) + \
                               0.3 * np.random.randn(*params[key].shape) * 0.01
        else:
            attacked = params.copy()
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attacked


def run_detection_experiment(attack_type, challenge_budget, num_rounds=20, 
                             malicious_client=0, device='cpu'):
    """Run detection experiment for specific attack type and challenge budget"""
    
    # Load data
    client_datasets = load_mnist_federated(num_clients=5, samples_per_client=10000)
    
    # Initialize models
    global_model = LeNet5().to(device)
    
    # Initialize RIV protocol
    riv = RIVProtocol(
        challenge_budget=challenge_budget,
        use_zero_knowledge=False,
        verbose=False
    )
    
    # Track detection results
    detections = []
    
    for round_idx in range(num_rounds):
        # Train all clients
        client_models = []
        client_gradients = []
        
        for client_id, dataset in enumerate(client_datasets):
            client_model = LeNet5().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            if client_id != malicious_client:
                # Honest client: train normally
                gradients = train_client(client_model, dataset, device=device)
            else:
                # Malicious client: may or may not train (depends on attack)
                gradients = train_client(client_model, dataset, device=device)
            
            client_models.append(client_model)
            client_gradients.append(gradients)
        
        # Extract parameters
        all_params = [extract_parameters(model) for model in client_models]
        
        # Apply attack to malicious client (get honest params for reference)
        honest_params = all_params[0] if malicious_client != 0 else all_params[1]
        all_params[malicious_client] = apply_attack(
            all_params[malicious_client],
            attack_type=attack_type,
            honest_params=honest_params
        )
        
        # RIV protocol for each client
        round_detected = False
        
        for client_id, params in enumerate(all_params):
            metadata = {
                'round': round_idx,
                'client_id': client_id,
                'learning_rate': 0.01
            }
            
            # Commitment phase
            commit_result = riv.client_commit_phase(params, metadata)
            
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
            
            # Verification phase
            verify_result = riv.server_verify_phase(
                commitments=commit_result['commitments'],
                proofs=proof_result['proofs'],
                challenged_layers=challenge_result['challenged_layers'],
                metadata=metadata
            )
            
            # Check if malicious client was detected
            if client_id == malicious_client and not verify_result['verified']:
                round_detected = True
        
        detections.append(1 if round_detected else 0)
    
    detection_rate = sum(detections) / len(detections)
    return detection_rate


def main():
    parser = argparse.ArgumentParser(description='Free-Rider Detection Experiment')
    parser.add_argument('--attack-types', nargs='+', default=['zero', 'random', 'stale', 'noise'],
                       help='Attack types to test')
    parser.add_argument('--challenge-budgets', nargs='+', type=int, default=[1, 3, 5, 7, 10],
                       help='Challenge budgets to test')
    parser.add_argument('--rounds', type=int, default=20, help='Rounds per experiment')
    parser.add_argument('--repetitions', type=int, default=5, help='Repetitions for averaging')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Free-Rider Detection Rate Experiment")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Attack types: {', '.join(args.attack_types)}")
    print(f"  Challenge budgets: {args.challenge_budgets}")
    print(f"  Rounds per experiment: {args.rounds}")
    print(f"  Repetitions: {args.repetitions}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Results storage
    results = {
        'config': vars(args),
        'detection_rates': {}
    }
    
    total_experiments = len(args.attack_types) * len(args.challenge_budgets) * args.repetitions
    experiment_count = 0
    
    print(f"Running {total_experiments} total experiments...")
    print()
    
    start_time = time.time()
    
    # Run experiments for each attack type and challenge budget
    for attack_type in args.attack_types:
        print(f"Testing {attack_type} attack...")
        results['detection_rates'][attack_type] = {}
        
        for k in args.challenge_budgets:
            detection_rates = []
            
            for rep in range(args.repetitions):
                experiment_count += 1
                print(f"  k={k}, repetition {rep+1}/{args.repetitions} " +
                      f"({experiment_count}/{total_experiments})", end='')
                
                rate = run_detection_experiment(
                    attack_type=attack_type,
                    challenge_budget=k,
                    num_rounds=args.rounds,
                    device=device
                )
                
                detection_rates.append(rate)
                print(f" -> {rate*100:.1f}% detected")
            
            avg_rate = np.mean(detection_rates)
            std_rate = np.std(detection_rates)
            
            results['detection_rates'][attack_type][f'k{k}'] = {
                'mean': avg_rate,
                'std': std_rate,
                'rates': detection_rates
            }
            
            print(f"  k={k} average: {avg_rate*100:.1f}% ± {std_rate*100:.1f}%")
        
        print()
    
    total_time = time.time() - start_time
    
    # Summary
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    
    print("Detection Rates Summary:")
    print()
    print(f"{'Attack':<12} " + " ".join([f"k={k:<3}" for k in args.challenge_budgets]))
    print("-" * 70)
    
    for attack_type in args.attack_types:
        rates = []
        for k in args.challenge_budgets:
            rate = results['detection_rates'][attack_type][f'k{k}']['mean']
            rates.append(f"{rate*100:>5.1f}%")
        print(f"{attack_type:<12} " + " ".join(rates))
    
    print()
    
    # Key findings
    print("Key Findings:")
    for attack_type in args.attack_types:
        k5_rate = results['detection_rates'][attack_type]['k5']['mean']
        print(f"  {attack_type}: {k5_rate*100:.1f}% detection at k=5")
    
    print()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'detection_rate_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    print("Next steps:")
    print("  - View results: cat", results_file)
    print("  - Test scalability: python experiments/scalability.py")
    print("  - Run full training: python experiments/lenet5_mnist.py")


if __name__ == '__main__':
    main()
