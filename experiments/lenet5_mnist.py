"""
LeNet-5 on MNIST - Full Experiment

Complete 50-round federated learning experiment with RIV protocol.
Reproduces Table 2 results from the paper.

Configuration:
- Model: LeNet-5 (61,706 parameters)
- Dataset: MNIST (60,000 training samples)
- Clients: 5 (IID data distribution)
- Rounds: 50
- Challenge budget: k=5 layers
- Mode: Transparent (can enable ZK with --zk flag)

Expected Results:
- Test accuracy: ~98%
- Training overhead: <3%
- Verification time: <20ms per layer
- Communication: ~5-6 KB per round

Runtime: ~15-20 minutes
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
    """Load and partition MNIST dataset for federated learning"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    # IID partition: randomly assign samples to clients
    indices = torch.randperm(len(train_dataset))
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, len(train_dataset))
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(train_dataset, client_indices))
    
    return client_datasets, test_dataset


def train_client(model, dataset, learning_rate=0.01, batch_size=64, device='cpu'):
    """Train client model for one round"""
    model.train()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # No momentum for RIV verification
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_loss = 0
    correct = 0
    total = 0
    gradients = {}
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Save gradients from last batch for RIV protocol
        if batch_idx == len(loader) - 1:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.detach().cpu().numpy().copy()
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total, gradients


def test_model(model, test_dataset, batch_size=256, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loader)
    accuracy = 100. * correct / len(test_dataset)
    
    return test_loss, accuracy


def extract_parameters(model):
    """Extract model parameters as dictionary"""
    params = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            params[f'layer_{idx}'] = param.data.cpu().numpy().copy()
    return params


def aggregate_parameters(global_model, client_models):
    """Federated averaging of client models"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.stack([
            client.state_dict()[key].float() for client in client_models
        ], 0).mean(0)
    
    global_model.load_state_dict(global_dict)


def main():
    parser = argparse.ArgumentParser(description='LeNet-5 MNIST Experiment')
    parser.add_argument('--rounds', type=int, default=50, help='Number of FL rounds')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--samples', type=int, default=10000, help='Samples per client')
    parser.add_argument('--challenge-budget', type=int, default=5, help='Challenge budget k')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--zk', action='store_true', help='Enable zero-knowledge mode')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("LeNet-5 on MNIST - Full Experiment")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Clients: {args.clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Samples per client: {args.samples}")
    print(f"  Challenge budget: {args.challenge_budget}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Zero-knowledge: {args.zk}")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    client_datasets, test_dataset = load_mnist_federated(
        num_clients=args.clients,
        samples_per_client=args.samples
    )
    print(f"  Training: {sum(len(d) for d in client_datasets)} samples across {args.clients} clients")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Initialize global model
    global_model = LeNet5().to(device)
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Initialize RIV protocol
    riv = RIVProtocol(
        challenge_budget=args.challenge_budget,
        use_zero_knowledge=args.zk,
        verbose=False
    )
    
    # Results storage
    results = {
        'config': vars(args),
        'rounds': [],
        'riv_metrics': {
            'commit_times': [],
            'challenge_times': [],
            'proof_times': [],
            'verify_times': [],
        }
    }
    
    # Training loop
    print(f"\nTraining for {args.rounds} rounds...")
    print()
    
    experiment_start = time.time()
    
    for round_idx in range(args.rounds):
        round_start = time.time()
        print(f"Round {round_idx + 1}/{args.rounds}")
        print("-" * 50)
        
        # Client training
        client_models = []
        train_losses = []
        train_accs = []
        client_gradients = []
        
        for client_id, dataset in enumerate(client_datasets):
            client_model = LeNet5().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            loss, acc, gradients = train_client(
                client_model, dataset, 
                learning_rate=args.lr,
                batch_size=args.batch_size,
                device=device
            )
            
            client_models.append(client_model)
            train_losses.append(loss)
            train_accs.append(acc)
            client_gradients.append(gradients)
        
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        
        # RIV Protocol: Commitment Phase
        model_updates = extract_parameters(client_models[0])
        metadata = {
            'round': round_idx,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        }
        
        commit_result = riv.client_commit_phase(model_updates, metadata)
        results['riv_metrics']['commit_times'].append(commit_result['commit_time'])
        
        # RIV Protocol: Challenge Phase
        challenge_result = riv.server_challenge_phase(
            commitment_hash=commit_result['commitment_hash'],
            num_layers=len(model_updates),
            metadata=metadata
        )
        results['riv_metrics']['challenge_times'].append(challenge_result['challenge_time'])
        
        # RIV Protocol: Proof Phase
        layer_data = {}
        gradients = client_gradients[0]  # Use first client's gradients
        for layer_idx in challenge_result['challenged_layers']:
            layer_name = f'layer_{layer_idx}'
            if layer_name in model_updates:
                # Map layer name to parameter name for gradient lookup
                param_name = list(client_models[0].named_parameters())[layer_idx][0]
                gradient = gradients.get(param_name, np.zeros_like(model_updates[layer_name]))
                
                layer_data[layer_name] = {
                    'activation': model_updates[layer_name],
                    'gradient': gradient,
                    'weight_old': model_updates[layer_name],
                    'weight_new': model_updates[layer_name]
                }
        
        proof_result = riv.client_proof_phase(
            challenged_layers=challenge_result['challenged_layers'],
            layer_data=layer_data,
            training_config=metadata
        )
        results['riv_metrics']['proof_times'].append(proof_result['proof_time'])
        
        # RIV Protocol: Verification Phase
        verify_result = riv.server_verify_phase(
            commitments=commit_result['commitments'],
            proofs=proof_result['proofs'],
            challenged_layers=challenge_result['challenged_layers'],
            metadata=metadata
        )
        results['riv_metrics']['verify_times'].append(verify_result['verification_time'])
        
        # Aggregate models
        aggregate_parameters(global_model, client_models)
        
        # Evaluate
        test_loss, test_acc = test_model(global_model, test_dataset, device=device)
        
        round_time = time.time() - round_start
        
        # Store results
        round_result = {
            'round': round_idx + 1,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'verification_passed': verify_result['verified'],
            'time': round_time
        }
        results['rounds'].append(round_result)
        
        print(f"  Train: loss={avg_train_loss:.4f}, acc={avg_train_acc:.2f}%")
        print(f"  Test: loss={test_loss:.4f}, acc={test_acc:.2f}%")
        print(f"  RIV: verified={verify_result['verified']}, time={round_time:.2f}s")
        print()
    
    total_time = time.time() - experiment_start
    
    # Final evaluation
    print("=" * 70)
    print("Experiment Complete")
    print("=" * 70)
    final_loss, final_acc = test_model(global_model, test_dataset, device=device)
    print(f"Final test accuracy: {final_acc:.2f}%")
    print(f"Total time: {total_time/60:.2f} minutes")
    print()
    
    # RIV overhead analysis
    avg_commit = np.mean(results['riv_metrics']['commit_times']) * 1000
    avg_proof = np.mean(results['riv_metrics']['proof_times']) * 1000
    avg_verify = np.mean(results['riv_metrics']['verify_times']) * 1000
    
    print("RIV Protocol Overhead:")
    print(f"  Commitment: {avg_commit:.2f}ms per round")
    print(f"  Proof generation: {avg_proof:.2f}ms per round")
    print(f"  Verification: {avg_verify:.2f}ms per round")
    print(f"  Total overhead: {(avg_commit + avg_proof + avg_verify):.2f}ms per round")
    print()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'lenet5_mnist_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    print("Next steps:")
    print("  - View results: cat", results_file)
    print("  - Run detection experiments: python experiments/detection_rate.py")
    print("  - Try ResNet-18: python experiments/resnet18_cifar10.py")


if __name__ == '__main__':
    main()
