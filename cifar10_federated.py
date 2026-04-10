"""
CIFAR-10 Federated Learning Pipeline
======================================
Implements:
  T5 - Real-Data Experiment with CIFAR-10 + FedAvg

Provides a realistic FL simulation with:
  - CIFAR-10 dataset
  - Simple CNN model
  - Dirichlet-based non-IID data splitting
  - FedAvg aggregation
  - Logit extraction for PINN Guard evaluation
  - Multiple poisoning attack integrations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from adversarial_pinn_guard import (
    train_adversarial_pinn_guard, get_violation_score, PINNGuard,
)
from fl_baselines import (
    get_all_baseline_defenses, PINNGuardDefense,
    shannon_entropy_score,
)


# ═══════════════════════════════════════════════════════════════
# CIFAR-10 Model
# ═══════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (before softmax)."""
        return self.forward(x)


# ═══════════════════════════════════════════════════════════════
# Data Partitioning (Non-IID via Dirichlet)
# ═══════════════════════════════════════════════════════════════

def dirichlet_split(
    dataset: torchvision.datasets.CIFAR10,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Split dataset indices among clients using Dirichlet distribution.
    
    Args:
        dataset: CIFAR-10 dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
    
    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)
    
    targets = np.array(dataset.targets)
    n_classes = len(set(targets))
    
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        
        # Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        
        # Fix rounding
        diff = len(class_indices) - proportions.sum()
        proportions[0] += diff
        
        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end
    
    return client_indices


# ═══════════════════════════════════════════════════════════════
# Federated Learning Simulation
# ═══════════════════════════════════════════════════════════════

class FederatedClient:
    """Represents a single FL client."""
    
    def __init__(
        self, client_id: int, dataset: torchvision.datasets.CIFAR10,
        indices: List[int], is_malicious: bool = False,
        target_class: int = 0, backdoor_label: int = 1,
        device: str = 'cpu',
    ):
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.target_class = target_class
        self.backdoor_label = backdoor_label
        self.device = device
        
        self.subset = Subset(dataset, indices)
        self.dataloader = DataLoader(self.subset, batch_size=32, shuffle=True)
    
    def train_local(
        self, global_model: SimpleCNN, n_epochs: int = 2, lr: float = 0.01,
    ) -> OrderedDict:
        """Train on local data and return model state dict."""
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_model.state_dict())
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(n_epochs):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.is_malicious:
                    # Backdoor: flip labels for target class
                    mask = labels == self.target_class
                    labels[mask] = self.backdoor_label
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def extract_logits(
        self, model: SimpleCNN, n_samples: int = 100,
    ) -> torch.Tensor:
        """Extract logits from the model on local data."""
        model.eval()
        all_logits = []
        count = 0
        
        with torch.no_grad():
            for inputs, _ in self.dataloader:
                inputs = inputs.to(self.device)
                logits = model.get_logits(inputs)
                all_logits.append(logits.cpu())
                count += inputs.size(0)
                if count >= n_samples:
                    break
        
        return torch.cat(all_logits, dim=0)[:n_samples]


class FedAvgServer:
    """FedAvg aggregation server with PINN Guard defense."""
    
    def __init__(self, n_classes: int = 10, device: str = 'cpu'):
        self.device = device
        self.global_model = SimpleCNN(n_classes).to(device)
        self.n_classes = n_classes
        self.pinn_guard = None
        self.defenses = []
        self.history = []
    
    def aggregate(self, client_updates: List[OrderedDict]) -> None:
        """FedAvg aggregation."""
        n = len(client_updates)
        avg_state = OrderedDict()
        
        for key in client_updates[0]:
            avg_state[key] = sum(
                update[key] for update in client_updates
            ) / n
        
        self.global_model.load_state_dict(avg_state)
    
    def train_pinn_guard(self, clean_logits: torch.Tensor, **kwargs):
        """Train the PINN Guard on clean logits."""
        self.pinn_guard, _ = train_adversarial_pinn_guard(
            clean_logits, device=self.device, **kwargs
        )
    
    def evaluate_client(
        self, client_logits: torch.Tensor,
    ) -> Dict[str, float]:
        """Score a client's logits with all defenses."""
        scores = {}
        
        if self.pinn_guard is not None:
            scores['PINN Guard'] = get_violation_score(
                self.pinn_guard, client_logits, self.device
            )
        
        for defense in self.defenses:
            scores[defense.name] = defense.score(client_logits)
        
        return scores
    
    def evaluate_accuracy(
        self, test_loader: DataLoader,
    ) -> float:
        """Evaluate global model accuracy on test set."""
        self.global_model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total


# ═══════════════════════════════════════════════════════════════
# T5: Full Pipeline
# ═══════════════════════════════════════════════════════════════

def run_cifar10_fl_experiment(
    n_clients: int = 20,
    n_malicious: int = 2,
    n_rounds: int = 30,
    alpha: float = 0.5,
    local_epochs: int = 2,
    device: str = 'cpu',
    output_dir: str = 'results/cifar10',
) -> pd.DataFrame:
    """
    T5: Run the full CIFAR-10 federated learning experiment.
    
    Args:
        n_clients: Total number of clients
        n_malicious: Number of malicious clients
        n_rounds: Number of FL rounds
        alpha: Dirichlet concentration for non-IID split
        local_epochs: Local training epochs per round
        device: Device
        output_dir: Output directory
    
    Returns:
        DataFrame with per-round results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"CIFAR-10 Federated Learning Experiment")
    print(f"  Clients: {n_clients} ({n_malicious} malicious)")
    print(f"  Rounds: {n_rounds}")
    print(f"  Non-IID α: {alpha}")
    print(f"  Device: {device}")
    print("=" * 60)
    
    # ── Load Data ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    
    # ── Split Data ──
    print("\n  Splitting data (Dirichlet non-IID)...")
    client_indices = dirichlet_split(trainset, n_clients, alpha)
    
    # ── Create Clients ──
    clients = []
    for i in range(n_clients):
        is_mal = i < n_malicious
        client = FederatedClient(
            client_id=i, dataset=trainset, indices=client_indices[i],
            is_malicious=is_mal, device=device,
        )
        clients.append(client)
        if is_mal:
            print(f"  Client {i}: MALICIOUS ({len(client_indices[i])} samples)")
        else:
            print(f"  Client {i}: Honest ({len(client_indices[i])} samples)")
    
    # ── Create Server ──
    server = FedAvgServer(n_classes=10, device=device)
    
    # ── Initialize Defenses ──
    print("\n  Initializing baseline defenses...")
    baselines = get_all_baseline_defenses()
    server.defenses = baselines
    
    # ── FL Training Loop ──
    all_results = []
    
    for round_idx in range(n_rounds):
        print(f"\n  Round {round_idx + 1}/{n_rounds}")
        
        # Local training
        updates = []
        client_logits_list = []
        
        for client in clients:
            update = client.train_local(server.global_model, n_epochs=local_epochs)
            updates.append(update)
            
            # Extract logits for defense evaluation
            temp_model = SimpleCNN().to(device)
            temp_model.load_state_dict(update)
            logits = client.extract_logits(temp_model, n_samples=100)
            client_logits_list.append(logits)
        
        # Train PINN Guard on clean logits (first honest client)
        if round_idx == 0:
            honest_logits = client_logits_list[n_malicious]  # First honest client
            print("  Training PINN Guard on clean logits...")
            server.train_pinn_guard(
                honest_logits, n_epochs=100, verbose=False
            )
            # Fit baseline defenses
            for d in baselines:
                d.fit(honest_logits)
        
        # Evaluate each client
        for i, (client, logits) in enumerate(zip(clients, client_logits_list)):
            scores = server.evaluate_client(logits)
            scores['Round'] = round_idx
            scores['Client_ID'] = i
            scores['Is_Malicious'] = client.is_malicious
            all_results.append(scores)
        
        # Aggregate (FedAvg)
        server.aggregate(updates)
        
        # Evaluate accuracy
        accuracy = server.evaluate_accuracy(test_loader)
        print(f"  Accuracy: {accuracy:.4f}")
    
    # ── Save Results ──
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'fl_results.csv'), index=False)
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("CIFAR-10 FL Experiment Complete")
    print("=" * 60)
    
    if 'PINN Guard' in df.columns:
        print("\nPINN Guard Detection Summary:")
        summary = df.groupby('Is_Malicious')['PINN Guard'].describe()
        print(summary)
    
    return df


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run with small config for quick testing
    df = run_cifar10_fl_experiment(
        n_clients=10,
        n_malicious=1,
        n_rounds=10,  # Increase to 50+ for paper
        alpha=0.5,
        device=device,
        output_dir='results/cifar10',
    )
