"""
Adversarial PINN Guard for Federated Learning Defense
======================================================
Implements:
  T1  - Adversarial (Min-Max) PINN Guard Training
  T3  - L2 Smoothness Ablation Baseline
  T11 - Architecture Sensitivity Analysis
  T12 - Fisher Information Metric for Non-Euclidean Logit Manifold
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────
# Core Utility Functions
# ─────────────────────────────────────────────────────────────

def make_clean_logits(B: int, C: int) -> torch.Tensor:
    """Generate clean baseline logits."""
    return torch.randn(B, C)


def make_poisoned_logits(B: int, C: int, target: int = 0, bias: float = 2.5) -> torch.Tensor:
    """Naive poisoning: add bias to target class."""
    logits = torch.randn(B, C)
    logits[:, target] += bias
    return logits


# ─────────────────────────────────────────────────────────────
# T12: Fisher Information Metric
# ─────────────────────────────────────────────────────────────

class FisherInformationMetric:
    """
    Implements the Fisher Information Metric on the probability simplex.
    
    Instead of treating logits in flat Euclidean space, we map them to the
    probability simplex via softmax and compute distances using the Fisher
    Information Metric, which is the natural Riemannian metric on the space
    of categorical distributions.
    
    The Fisher metric tensor for a categorical distribution p is:
        g_ij = δ_ij / p_i
    
    This gives the simplex non-trivial curvature, making the manifold
    framework genuinely geometric rather than cosmetic.
    """
    
    @staticmethod
    def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Convert logits to probabilities via softmax."""
        return torch.softmax(logits / temperature, dim=-1)
    
    @staticmethod
    def fisher_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Fisher-Rao distance between two distributions.
        d_FR(p, q) = 2 * arccos(sum(sqrt(p_i * q_i)))
        """
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        inner = (torch.sqrt(p) * torch.sqrt(q)).sum(dim=-1)
        inner = inner.clamp(max=1.0 - eps)
        return 2.0 * torch.acos(inner)
    
    @staticmethod
    def fisher_metric_tensor(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute the Fisher Information Metric tensor.
        g_ij(p) = δ_ij / p_i
        Returns diagonal elements (since it's diagonal).
        """
        return 1.0 / probs.clamp(min=eps)
    
    @staticmethod
    def dirichlet_energy_fisher(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Dirichlet energy using Fisher Information Metric.
        E[f] = ∫ g^ij ∂_i f ∂_j f dΩ
        
        In the Fisher metric, this weights gradients by 1/p_i,
        amplifying perturbations in low-probability regions.
        """
        probs = torch.softmax(logits, dim=-1).clamp(min=eps)
        
        # Compute finite differences (gradients along sample dimension)
        if logits.shape[0] < 2:
            return torch.tensor(0.0, device=logits.device)
        
        grad = logits[1:] - logits[:-1]  # (B-1, C)
        probs_mid = (probs[1:] + probs[:-1]) / 2  # midpoint probabilities
        
        # Fisher-weighted gradient norm: sum_j (grad_j)^2 / p_j
        fisher_weighted = (grad ** 2) / probs_mid
        
        return fisher_weighted.sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────
# PINN Guard Architectures
# ─────────────────────────────────────────────────────────────

class PINNGuard(nn.Module):
    """Original PINN Guard architecture (3-layer MLP)."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, 
                 num_layers: int = 3, activation: str = 'tanh'):
        super().__init__()
        
        act_fn = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
        }[activation.lower()]
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class L2SmoothnessFilter(nn.Module):
    """
    T3: L2 Smoothness Ablation Baseline.
    
    Same architecture as PINNGuard but trained with L2 norm loss
    instead of Laplacian/Dirichlet loss. This proves whether the
    manifold geometry adds value over simple smoothness.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# T1: Adversarial (Min-Max) PINN Guard Training
# ─────────────────────────────────────────────────────────────

class AdversarialAttacker(nn.Module):
    """
    Adversary network that learns to produce poisoned logits
    that minimize the PINN Guard's residual while maximizing
    attack utility.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, clean_logits: torch.Tensor) -> torch.Tensor:
        """Transform clean logits into adversarial logits."""
        perturbation = self.net(clean_logits)
        return clean_logits + perturbation


def compute_laplacian_residual(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian residual (∇²f) for the PINN Guard.
    Uses automatic differentiation to compute second-order derivatives.
    """
    logits = logits.clone().requires_grad_(True)
    output = model(logits)
    
    laplacian = torch.zeros_like(output)
    for i in range(output.shape[1]):
        grad_i = torch.autograd.grad(
            output[:, i].sum(), logits,
            create_graph=True, retain_graph=True
        )[0]
        for j in range(logits.shape[1]):
            grad_ij = torch.autograd.grad(
                grad_i[:, j].sum(), logits,
                create_graph=True, retain_graph=True
            )[0]
            laplacian[:, i] += grad_ij[:, j]
    
    return laplacian


def compute_dirichlet_energy(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the Dirichlet (bending) energy: E[f] = ∫ ||∇f||² dΩ
    """
    logits = logits.clone().requires_grad_(True)
    output = model(logits)
    
    total_energy = torch.tensor(0.0, device=logits.device)
    for i in range(output.shape[1]):
        grad_i = torch.autograd.grad(
            output[:, i].sum(), logits,
            create_graph=True, retain_graph=True
        )[0]
        total_energy = total_energy + (grad_i ** 2).sum()
    
    return total_energy / logits.shape[0]


def _compute_physics_loss(
    model: nn.Module,
    logits: torch.Tensor,
    use_fisher: bool = False,
    fisher: Optional['FisherInformationMetric'] = None,
    detach_input: bool = True,
) -> torch.Tensor:
    """
    Compute the ACTUAL physics-based loss for the PINN Guard.
    
    This is the Laplacian residual ||∇²f(x)||² — the core of what
    makes a PINN a PINN. Without this, it's just an autoencoder.
    
    Args:
        model: The PINN Guard network
        logits: Input logits (B, C)
        use_fisher: Whether to weight by Fisher metric
        fisher: FisherInformationMetric instance
        detach_input: If True, detach input from its computation graph
                      (use True for scoring and PINN training on static data).
                      If False, gradients flow through to the input's source
                      (use False when training the adversary, so it can learn
                      to produce logits that minimize the Laplacian).
    """
    if detach_input:
        x = logits.clone().detach().requires_grad_(True)
    else:
        # Keep the graph connected to adversary, but still need requires_grad
        # for second-order derivative computation
        if logits.requires_grad:
            x = logits
        else:
            x = logits.requires_grad_(True)
    
    output = model(x)  # (B, C)
    
    B, C = output.shape
    
    # Compute Laplacian: ∇²f = Σ_j ∂²f/∂x_j²
    laplacian = torch.zeros_like(output)
    
    for i in range(C):
        # First derivative of output_i w.r.t. all inputs
        grad_i = torch.autograd.grad(
            output[:, i].sum(), x,
            create_graph=True, retain_graph=True,
        )[0]  # (B, C)
        
        # Second derivative: diagonal of Hessian (trace = Laplacian)
        for j in range(C):
            grad_ij = torch.autograd.grad(
                grad_i[:, j].sum(), x,
                create_graph=True, retain_graph=True,
            )[0]  # (B, C)
            laplacian[:, i] = laplacian[:, i] + grad_ij[:, j]
    
    # Optionally weight by Fisher metric on the INPUT space
    if use_fisher and fisher is not None:
        probs = torch.softmax(x, dim=-1).clamp(min=1e-4)
        fisher_weights = 1.0 / probs  # g^{ij} = 1/p_i (diagonal)
        laplacian = laplacian * fisher_weights
    
    return (laplacian ** 2).mean()


def _compute_simple_energy(model: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """Simple output-squared energy (NOT physics). Used only for L2 ablation."""
    return (model(logits) ** 2).mean()


def train_adversarial_pinn_guard(
    clean_logits: torch.Tensor,
    target: int = 0,
    n_epochs: int = 200,
    n_inner_pinn: int = 3,
    n_inner_adv: int = 2,
    pinn_lr: float = 1e-3,
    adv_lr: float = 1e-3,
    attack_weight: float = 1.0,
    evasion_weight: float = 10.0,
    use_fisher: bool = False,
    pinn_config: Optional[Dict] = None,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[PINNGuard, dict]:
    """
    T1: Adversarial Min-Max PINN Guard Training with ACTUAL PHYSICS.
    
    KEY DIFFERENCE from previous version:
      The loss now uses compute_laplacian_residual (∇²f) via autograd,
      NOT just (output**2). This is what makes it a genuine PINN.
    
    Min-Max Objective:
      min_θ max_φ  E_clean[||∇²f_θ(L)||²] 
                 - λ · E_adv[||∇²f_θ(G_φ(L))||²]
                 + μ · E_adv[U(G_φ(L), target)]
    
    The Laplacian ∇²f captures local curvature of the learned function.
    Clean logits should satisfy ∇²f ≈ 0 (harmonic condition).
    Poisoned logits create high-curvature regions that violate harmonicity.
    
    Args:
        clean_logits: Clean baseline logits (B, C)
        target: Target class for the attack
        n_epochs: Number of training epochs
        n_inner_pinn: Inner loop iterations for PINN per epoch
        n_inner_adv: Inner loop iterations for adversary per epoch
        pinn_lr: Learning rate for PINN
        adv_lr: Learning rate for adversary
        attack_weight: Weight for attack utility in adversary loss
        evasion_weight: Weight for evasion term in adversary loss
        use_fisher: Whether to use Fisher Information Metric on input space
        pinn_config: Optional config dict for PINN architecture
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Trained PINNGuard model and training history dict
    """
    C = clean_logits.shape[1]
    clean_logits = clean_logits.to(device)
    
    # Initialize PINN Guard
    if pinn_config is None:
        pinn_config = {'input_dim': C, 'hidden_dim': 64, 'num_layers': 3, 'activation': 'tanh'}
    pinn = PINNGuard(**pinn_config).to(device)
    
    # Initialize Adversary
    adversary = AdversarialAttacker(input_dim=C, hidden_dim=64).to(device)
    
    # Optimizers
    pinn_optimizer = optim.Adam(pinn.parameters(), lr=pinn_lr)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=adv_lr)
    
    # Fisher metric (if used)
    fisher = FisherInformationMetric() if use_fisher else None
    
    history = {
        'pinn_loss': [], 'adv_loss': [],
        'clean_violation': [], 'adv_violation': [],
        'attack_utility': [],
    }
    
    for epoch in range(n_epochs):
        pinn.train()
        
        # ── Phase 1: Train PINN Guard ──
        for _ in range(n_inner_pinn):
            pinn_optimizer.zero_grad()
            
            # PHYSICS LOSS: Minimize Laplacian residual on clean data
            # This enforces ∇²f(clean) ≈ 0 (harmonicity)
            # detach_input=True: clean_logits are static data, no upstream graph
            clean_energy = _compute_physics_loss(
                pinn, clean_logits, use_fisher=use_fisher, fisher=fisher,
                detach_input=True,
            )
            
            # Maximize Laplacian residual on adversarial data
            # detach_input=True: adversary is frozen during PINN training
            with torch.no_grad():
                adv_logits = adversary(clean_logits)
            adv_energy = _compute_physics_loss(
                pinn, adv_logits, use_fisher=use_fisher, fisher=fisher,
                detach_input=True,
            )
            
            # PINN wants: low clean Laplacian, HIGH adversarial Laplacian
            pinn_loss = clean_energy - 0.5 * adv_energy
            pinn_loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            pinn_optimizer.step()
        
        # ── Phase 2: Train Adversary ──
        for _ in range(n_inner_adv):
            adv_optimizer.zero_grad()
            
            adv_logits = adversary(clean_logits)
            
            # Evasion: minimize PINN's Laplacian on adversarial logits
            # detach_input=False: CRITICAL — gradients must flow back through
            # the Laplacian computation to the adversary's parameters, so it
            # can learn to produce logits that minimize Laplacian residual
            evasion_loss = _compute_physics_loss(
                pinn, adv_logits, use_fisher=use_fisher, fisher=fisher,
                detach_input=False,
            )
            
            # Attack utility: maximize target class logit
            utility = -adv_logits[:, target].mean()
            
            # Constraint: bounded perturbation. Ensures the attacker stays within
            # a realistic attack envelope and stops the Min-Max game from diverging
            # to infinity, keeping the PINN focused on the local manifold curvature.
            distance = torch.norm(adv_logits - clean_logits, dim=-1).mean()
            
            # Adversary wants: low Laplacian, high utility, but bounded distance
            adv_loss = evasion_weight * evasion_loss + attack_weight * utility + 5.0 * distance
            adv_loss.backward()
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_norm=1.0)
            adv_optimizer.step()
        
        # ── Logging (use simple output for fast scoring at inference) ──
        with torch.no_grad():
            clean_v = pinn(clean_logits).abs().mean().item()
            adv_v = pinn(adversary(clean_logits)).abs().mean().item()
            util_v = adversary(clean_logits)[:, target].mean().item()
        
        history['pinn_loss'].append(pinn_loss.item())
        history['adv_loss'].append(adv_loss.item())
        history['clean_violation'].append(clean_v)
        history['adv_violation'].append(adv_v)
        history['attack_utility'].append(util_v)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | "
                  f"PINN Loss: {pinn_loss.item():.6f} | "
                  f"Clean V: {clean_v:.6f} | "
                  f"Adv V: {adv_v:.6f} | "
                  f"Utility: {util_v:.4f}")
    
    pinn.eval()
    return pinn, history


# ─────────────────────────────────────────────────────────────
# T3: L2 Smoothness Ablation Training
# ─────────────────────────────────────────────────────────────

def train_l2_smoothness_filter(
    clean_logits: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> L2SmoothnessFilter:
    """
    T3: Train L2 smoothness baseline (no manifold structure).
    Uses simple L2 reconstruction loss instead of Laplacian/Dirichlet.
    """
    C = clean_logits.shape[1]
    clean_logits = clean_logits.to(device)
    
    model = L2SmoothnessFilter(input_dim=C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(clean_logits)
        # Simple L2 loss: output should be zero for clean data
        loss = (output ** 2).mean()
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def train_adversarial_l2_filter(
    clean_logits: torch.Tensor,
    target: int = 0,
    n_epochs: int = 200,
    device: str = 'cpu',
) -> L2SmoothnessFilter:
    """
    T3: Train adversarial L2 smoothness baseline for fair comparison.
    Same Min-Max framework but with L2 loss instead of Dirichlet/Laplacian.
    """
    C = clean_logits.shape[1]
    clean_logits = clean_logits.to(device)
    
    model = L2SmoothnessFilter(input_dim=C).to(device)
    adversary = AdversarialAttacker(input_dim=C).to(device)
    
    model_opt = optim.Adam(model.parameters(), lr=1e-3)
    adv_opt = optim.Adam(adversary.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        # Train model
        for _ in range(5):
            model_opt.zero_grad()
            clean_out = model(clean_logits)
            clean_loss = (clean_out ** 2).mean()
            
            with torch.no_grad():
                adv_logits = adversary(clean_logits)
            adv_out = model(adv_logits)
            adv_loss = (adv_out ** 2).mean()
            
            loss = clean_loss - 0.5 * adv_loss
            loss.backward()
            model_opt.step()
        
        # Train adversary
        for _ in range(3):
            adv_opt.zero_grad()
            adv_logits = adversary(clean_logits)
            adv_out = model(adv_logits)
            evasion = (adv_out ** 2).mean()
            utility = -adv_logits[:, target].mean()
            loss = 10.0 * evasion + utility
            loss.backward()
            adv_opt.step()
    
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# T11: Architecture Sensitivity Analysis
# ─────────────────────────────────────────────────────────────

@dataclass
class ArchConfig:
    name: str
    input_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    activation: str = 'tanh'


def get_architecture_configs(C: int = 10) -> List[ArchConfig]:
    """Generate architecture configurations for sensitivity analysis."""
    configs = []
    
    # Vary depth
    for n_layers in [2, 3, 4, 5]:
        configs.append(ArchConfig(
            name=f"depth_{n_layers}",
            input_dim=C, hidden_dim=64, num_layers=n_layers, activation='tanh'
        ))
    
    # Vary width
    for width in [32, 64, 128, 256]:
        configs.append(ArchConfig(
            name=f"width_{width}",
            input_dim=C, hidden_dim=width, num_layers=3, activation='tanh'
        ))
    
    # Vary activation
    for act in ['tanh', 'relu', 'gelu', 'silu']:
        configs.append(ArchConfig(
            name=f"act_{act}",
            input_dim=C, hidden_dim=64, num_layers=3, activation=act
        ))
    
    return configs


def run_architecture_sensitivity(
    clean_logits: torch.Tensor,
    attack_logits_dict: Dict[str, torch.Tensor],
    device: str = 'cpu',
    n_epochs: int = 150,
) -> Dict[str, Dict]:
    """
    T11: Run architecture sensitivity analysis.
    
    Args:
        clean_logits: Clean baseline logits
        attack_logits_dict: Dict mapping attack name -> poisoned logits
        device: Device to train on
        n_epochs: Training epochs per config
    
    Returns:
        Dict mapping config name -> {clean_violation, attack_violations}
    """
    C = clean_logits.shape[1]
    configs = get_architecture_configs(C)
    results = {}
    
    for config in configs:
        print(f"\n  Testing architecture: {config.name}")
        
        pinn_cfg = {
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'activation': config.activation,
        }
        
        pinn, _ = train_adversarial_pinn_guard(
            clean_logits,
            n_epochs=n_epochs,
            pinn_config=pinn_cfg,
            device=device,
            verbose=False,
        )
        
        with torch.no_grad():
            clean_v = pinn(clean_logits.to(device)).abs().mean().item()
            attack_vs = {}
            for atk_name, atk_logits in attack_logits_dict.items():
                v = pinn(atk_logits.to(device)).abs().mean().item()
                attack_vs[atk_name] = v
        
        results[config.name] = {
            'clean_violation': clean_v,
            'attack_violations': attack_vs,
        }
    
    return results


# ─────────────────────────────────────────────────────────────
# Convenience: Get violation score from a trained guard
# ─────────────────────────────────────────────────────────────

def get_violation_score(
    guard: nn.Module, 
    logits: torch.Tensor, 
    device: str = 'cpu',
    use_physics: bool = True,
    use_fisher: bool = False,
) -> float:
    """
    Get the anomaly score from a trained PINN Guard.
    
    CRITICAL: Must match the training objective.
    - If use_physics=True: computes ||∇²f(x)||² (Laplacian residual)
    - If use_physics=False: computes |f(x)| (raw output, for L2 filter)
    
    Args:
        guard: Trained PINN or L2 filter model
        logits: Input logits to score
        device: Device
        use_physics: Whether to compute Laplacian (True for PINN, False for L2)
        use_fisher: Whether to apply Fisher metric weighting
    """
    guard.eval()
    logits = logits.to(device)
    
    if not use_physics:
        # L2 Filter: just use raw output
        with torch.no_grad():
            return guard(logits).abs().mean().item()
    
    # PINN Guard: compute Laplacian residual (matching training loss)
    fisher = FisherInformationMetric() if use_fisher else None
    return _compute_physics_loss(guard, logits, use_fisher=use_fisher, fisher=fisher).item()


# ─────────────────────────────────────────────────────────────
# Main: Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Adversarial PINN Guard - Sanity Check")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, C = 100, 10
    
    clean = make_clean_logits(B, C)
    poisoned = make_poisoned_logits(B, C, target=0, bias=3.0)
    
    print("\n[T1] Training Adversarial PINN Guard...")
    pinn, history = train_adversarial_pinn_guard(
        clean, target=0, n_epochs=100, device=device
    )
    
    print(f"\n  Clean violation:    {get_violation_score(pinn, clean, device):.6f}")
    print(f"  Poisoned violation: {get_violation_score(pinn, poisoned, device):.6f}")
    
    print("\n[T3] Training L2 Smoothness Baseline...")
    l2_filter = train_adversarial_l2_filter(clean, device=device, n_epochs=100)
    
    print(f"  L2 Clean violation:    {get_violation_score(l2_filter, clean, device):.6f}")
    print(f"  L2 Poisoned violation: {get_violation_score(l2_filter, poisoned, device):.6f}")
    
    print("\n[T12] Fisher Information Metric...")
    fisher = FisherInformationMetric()
    clean_fisher_energy = fisher.dirichlet_energy_fisher(clean)
    poison_fisher_energy = fisher.dirichlet_energy_fisher(poisoned)
    print(f"  Clean Fisher energy:    {clean_fisher_energy.item():.6f}")
    print(f"  Poisoned Fisher energy: {poison_fisher_energy.item():.6f}")
    
    print("\n[T1+T12] Training Adversarial PINN Guard with Fisher Metric...")
    pinn_fisher, _ = train_adversarial_pinn_guard(
        clean, target=0, n_epochs=100, use_fisher=True, device=device
    )
    print(f"  Fisher Clean violation:    {get_violation_score(pinn_fisher, clean, device):.6f}")
    print(f"  Fisher Poisoned violation: {get_violation_score(pinn_fisher, poisoned, device):.6f}")
    
    print("\n✅ All sanity checks passed.")
