"""
LeNet-5 on MNIST - Quick Demo

A quick demonstration of the RIV protocol with LeNet-5 on MNIST.
This is a minimal example to verify the installation and show basic functionality.

Runtime: ~5 minutes
Output: Training progress, verification results, basic metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from riv_protocol import RIVProtocol


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network.
    
    Architecture:
    - Conv1: 1→6 channels, 5×5 kernel
    - Pool1: 2×2 max pooling
    - Conv2: 6→16 channels, 5×5 kernel
    - Pool2: 2×2 max pooling
    - FC1: 256→120
    - FC2: 120→84
    - FC3: 84→10 (output)
    
    Total parameters: ~61,000
    """
    
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


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        '../../data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        '../../data/mnist',
        train=False,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_one_round(model, train_loader, optimizer, device):
    """Train for one round (limited batches for demo)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Limit to a few batches for quick demo
    max_batches = 50
    
    # Will capture proof data from FIRST batch (fresh gradients)
    proof_captured = False
    gradients = {}
    weights_before = {}
    weights_after = {}
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        
        data, target = data.to(device), target.to(device)
        
        # Capture BEFORE weights on first batch
        if batch_idx == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights_before[name] = param.data.detach().cpu().numpy().copy()
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Capture gradients on first batch
        if batch_idx == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.detach().cpu().numpy().copy()
        
        optimizer.step()
        
        # Capture AFTER weights on first batch
        if batch_idx == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weights_after[name] = param.data.detach().cpu().numpy().copy()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / min(max_batches, len(train_loader))
    
    return avg_loss, accuracy, gradients, weights_before, weights_after


def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def extract_model_updates(model):
    """Extract model parameters as dictionary."""
    updates = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            updates[f'layer_{idx}'] = param.data.cpu().numpy().copy()
    return updates


def main():
    print("=" * 60)
    print("RIV Protocol - LeNet-5 MNIST Demo")
    print("=" * 60)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Initialize model
    print("Initializing LeNet-5...")
    model = LeNet5().to(device)
    # Note: No momentum for RIV gradient verification (momentum requires state tracking)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Initialize RIV protocol (transparent mode for demo)
    print("Initializing RIV protocol...")
    riv = RIVProtocol(
        challenge_budget=3,  # Challenge 3 layers
        use_zero_knowledge=False,  # Transparent mode (faster)
        verbose=True
    )
    print()
    
    # Training loop
    num_rounds = 5
    print(f"Training for {num_rounds} rounds...")
    print()
    
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}/{num_rounds}")
        print("-" * 40)
        
        # RIV Protocol: Commitment Phase (BEFORE training)
        # Commit to model state BEFORE this round's training
        model_before = extract_model_updates(model)
        metadata = {
            'round': round_idx,
            'learning_rate': 0.01,
            'batch_size': 64
        }
        
        commit_result = riv.client_commit_phase(model_before, metadata)
        print(f"  Commitment: {len(commit_result['commitments'])} layers, {commit_result['commit_time']:.3f}s")
        
        # Train (this modifies the model)
        round_start = time.time()
        train_loss, train_acc, gradients, weights_before, weights_after = train_one_round(model, train_loader, optimizer, device)
        train_time = time.time() - round_start
        
        print(f"  Training: loss={train_loss:.4f}, acc={train_acc:.2f}%, time={train_time:.2f}s")
        
        # Model after training
        model_after = extract_model_updates(model)
        
        # RIV Protocol: Challenge Phase
        challenge_result = riv.server_challenge_phase(
            commitment_hash=commit_result['commitment_hash'],
            num_layers=len(model_before),
            metadata=metadata
        )
        print(f"  Challenge: {len(challenge_result['challenged_layers'])} layers selected")
        
        # RIV Protocol: Proof Phase
        # Prove that first batch update was computed correctly
        layer_data = {}
        for layer_idx in challenge_result['challenged_layers']:
            layer_name = f'layer_{layer_idx}'
            if layer_name in model_before:
                # Map layer name to parameter name for gradient lookup
                param_name = list(model.named_parameters())[layer_idx][0]
                gradient = gradients.get(param_name, np.zeros_like(model_before[layer_name]))
                
                # weight_old should be what we committed to (model_before)
                # weight_new should be after first batch (weights_after)
                layer_data[layer_name] = {
                    'activation': model_before[layer_name],  # Committed value (before training)
                    'gradient': gradient,  # From first batch
                    'weight_old': model_before[layer_name],  # Same as committed (before training)
                    'weight_new': weights_after.get(param_name, model_before[layer_name])  # After first batch
                }
        
        proof_result = riv.client_proof_phase(
            challenged_layers=challenge_result['challenged_layers'],
            layer_data=layer_data,
            training_config=metadata
        )
        print(f"  Proof generation: {proof_result['proof_time']:.3f}s")
        
        # RIV Protocol: Verification Phase
        verify_result = riv.server_verify_phase(
            commitments=commit_result['commitments'],
            proofs=proof_result['proofs'],
            challenged_layers=challenge_result['challenged_layers'],
            metadata=metadata
        )
        
        status = "PASS" if verify_result['verified'] else "FAIL"
        print(f"  Verification: {status}, {verify_result['verification_time']:.3f}s")
        
        # Evaluate
        if (round_idx + 1) % 5 == 0:
            test_acc = test_model(model, test_loader, device)
            print(f"  Test accuracy: {test_acc:.2f}%")
        
        print()
    
    # Final evaluation
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    final_acc = test_model(model, test_loader, device)
    print(f"Final test accuracy: {final_acc:.2f}%")
    print()
    
    # Summary
    metrics = riv.get_metrics()
    print("RIV Protocol Metrics:")
    print(f"  Challenge budget (k): {metrics['challenge_budget']}")
    print(f"  Zero-knowledge mode: {metrics['zero_knowledge']}")
    print(f"  Total commitments: {metrics['num_commitments']}")
    print()
    
    print("Demo complete!")
    print()
    print("Next steps:")
    print("  - Run full experiments: python experiments/lenet5_mnist.py")
    print("  - Try ResNet-18: python experiments/resnet18_cifar10.py")
    print("  - Detection experiments: python experiments/detection_rate.py")


if __name__ == '__main__':
    main()
