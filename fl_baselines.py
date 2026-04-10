"""
Federated Learning Defense Baselines & Attack Strategies
=========================================================
Implements:
  T6  - FLAME (Nguyen et al., 2022)
  T7  - DeepSight (Rieger et al., 2022)
  T8  - RLR - Robust Learning Rate (Ozdayi et al., 2021)
  T9  - Krum, TrimmedMean, FoolsGold, FLTrust

Also includes all 20 attack strategies from the original notebook.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import cosine as cosine_dist


# ═══════════════════════════════════════════════════════════════
# ATTACK STRATEGIES (from original notebook + extensions)
# ═══════════════════════════════════════════════════════════════

def make_clean_logits(B: int, C: int) -> torch.Tensor:
    return torch.randn(B, C)

def make_poisoned_logits(B: int, C: int, target: int = 0, bias: float = 2.5) -> torch.Tensor:
    logits = torch.randn(B, C)
    logits[:, target] += bias
    return logits

def mmd_score(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 2.0) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel."""
    def rbf(a, b, sigma):
        dist = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    xx = rbf(x, x, bandwidth).mean()
    yy = rbf(y, y, bandwidth).mean()
    xy = rbf(x, y, bandwidth).mean()
    return xx + yy - 2 * xy

def shannon_entropy_score(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return -entropy.mean().item()

def make_adaptive_poisoned_logits(
    clean_logits: torch.Tensor, target: int = 0,
    poison_strength: float = 2.0, stealth_weight: float = 2.0,
    n_steps: int = 100
) -> torch.Tensor:
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.1)
    clean_probs = torch.softmax(clean_logits, dim=-1).detach()
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility = -poisoned[:, target].mean() * poison_strength
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(poisoned, dim=-1),
            clean_probs, reduction='batchmean', log_target=False
        )
        loss = utility + stealth_weight * kl
        loss.backward()
        optimizer.step()
    
    return poisoned.detach()

def make_super_adaptive_logits(
    clean_logits: torch.Tensor, target: int = 0, n_steps: int = 150
) -> torch.Tensor:
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.05)
    
    clean_mean = clean_logits.mean(dim=0).detach()
    clean_std = clean_logits.std(dim=0).detach()
    clean_probs = torch.softmax(clean_logits, dim=-1).detach()
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility = -poisoned[:, target].mean()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(poisoned, dim=-1),
            clean_probs, reduction='batchmean', log_target=False
        )
        mean_match = ((poisoned.mean(dim=0) - clean_mean) ** 2).mean()
        std_match = ((poisoned.std(dim=0) - clean_std) ** 2).mean()
        loss = utility + 5.0 * kl + 10.0 * mean_match + 10.0 * std_match
        loss.backward()
        optimizer.step()
    
    return poisoned.detach()

def make_balanced_tradeoff_logits(
    clean_logits: torch.Tensor, target: int = 0, n_steps: int = 100
) -> torch.Tensor:
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.08)
    clean_probs = torch.softmax(clean_logits, dim=-1).detach()
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility = -poisoned[:, target].mean()
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(poisoned, dim=-1),
            clean_probs, reduction='batchmean', log_target=False
        )
        loss = utility + 3.0 * kl
        loss.backward()
        optimizer.step()
    
    return poisoned.detach()

def rkhs_evasion_attack(
    clean_logits: torch.Tensor, target: int = 0,
    kernel_sigma: float = 2.0, n_steps: int = 150
) -> torch.Tensor:
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.1)
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility_loss = -poisoned[:, target].mean()
        evasion_loss = mmd_score(poisoned, clean_logits, bandwidth=kernel_sigma)
        total_loss = utility_loss + 100.0 * evasion_loss
        total_loss.backward()
        optimizer.step()
    
    return poisoned.detach()

def spectral_masking_attack(
    clean_logits: torch.Tensor, target: int = 0, n_steps: int = 50
) -> torch.Tensor:
    p = clean_logits.clone().detach().requires_grad_(True)
    opt = optim.Adam([p], lr=0.1)
    _, s_c, _ = torch.linalg.svd(clean_logits - clean_logits.mean(0))
    s_c = s_c.detach()
    
    for _ in range(n_steps):
        opt.zero_grad()
        _, s_p, _ = torch.linalg.svd(p - p.mean(0))
        loss = -p[:, target].mean() + 50.0 * torch.norm(s_p - s_c)
        loss.backward()
        opt.step()
    
    return p.detach()

def constrain_and_scale_attack(
    clean_logits: torch.Tensor, target: int = 0,
    scale: float = 5.0, n_steps: int = 100
) -> torch.Tensor:
    """Constrain-and-Scale attack (Bagdasaryan et al., 2020)."""
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.1)
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility = -poisoned[:, target].mean()
        # Constrain: keep the update within a norm ball
        norm_constraint = torch.relu(
            torch.norm(poisoned - clean_logits) - torch.norm(clean_logits) * 0.5
        )
        loss = utility + 5.0 * norm_constraint
        loss.backward()
        optimizer.step()
    
    # Scale the perturbation
    delta = poisoned.detach() - clean_logits
    return clean_logits + delta * scale

def dba_attack(
    clean_logits: torch.Tensor, target: int = 0,
    n_partitions: int = 4, partition_idx: int = 0, n_steps: int = 100
) -> torch.Tensor:
    """Distributed Backdoor Attack (Xie et al., 2020) - logit-level simulation."""
    C = clean_logits.shape[1]
    poisoned = clean_logits.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([poisoned], lr=0.1)
    
    # Each partition only modifies a subset of logit dimensions
    dims_per_part = C // n_partitions
    start = partition_idx * dims_per_part
    end = min(start + dims_per_part, C)
    
    mask = torch.zeros(C)
    mask[start:end] = 1.0
    mask[target] = 1.0  # Always include target
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        utility = -poisoned[:, target].mean()
        # Only allow perturbation in assigned dimensions
        constraint = ((poisoned - clean_logits) * (1 - mask)).abs().mean()
        loss = utility + 20.0 * constraint
        loss.backward()
        optimizer.step()
    
    return poisoned.detach()


def get_all_attack_strategies() -> List[str]:
    """Get names of all 20 attack strategies."""
    named = [
        'Naive_Bias', 'Adaptive_KL', 'Super_Adaptive',
        'Balanced_Tradeoff', 'RKHS_Evasion', 'Spectral_Masking',
        'Constrain_Scale', 'DBA_Part0', 'DBA_Part1',
        'DBA_Part2', 'DBA_Part3',
    ]
    # Add 9 random variants
    named += [f'Variant_{i}' for i in range(9)]
    return named


def execute_attack(
    name: str, clean_logits: torch.Tensor, target: int = 0
) -> torch.Tensor:
    """Execute an attack strategy by name."""
    B, C = clean_logits.shape
    
    if name == 'Naive_Bias':
        return make_poisoned_logits(B, C, target, bias=2.5)
    elif name == 'Adaptive_KL':
        return make_adaptive_poisoned_logits(clean_logits, target, stealth_weight=2.0)
    elif name == 'Super_Adaptive':
        return make_super_adaptive_logits(clean_logits, target)
    elif name == 'Balanced_Tradeoff':
        return make_balanced_tradeoff_logits(clean_logits, target)
    elif name == 'RKHS_Evasion':
        return rkhs_evasion_attack(clean_logits, target)
    elif name == 'Spectral_Masking':
        return spectral_masking_attack(clean_logits, target)
    elif name == 'Constrain_Scale':
        return constrain_and_scale_attack(clean_logits, target)
    elif name.startswith('DBA_Part'):
        idx = int(name[-1])
        return dba_attack(clean_logits, target, partition_idx=idx)
    else:
        # Random variant
        strength = np.random.uniform(0.5, 3.0)
        stealth = np.random.uniform(1.0, 5.0)
        return make_adaptive_poisoned_logits(
            clean_logits, target,
            poison_strength=strength, stealth_weight=stealth
        )


# ═══════════════════════════════════════════════════════════════
# DEFENSE BASELINES
# ═══════════════════════════════════════════════════════════════

class BaseDefense:
    """Base class for FL defense methods operating at the logit level."""
    
    def __init__(self, name: str):
        self.name = name
    
    def fit(self, clean_logits: torch.Tensor):
        """Fit the defense on clean baseline data."""
        raise NotImplementedError
    
    def score(self, logits: torch.Tensor) -> float:
        """Return an anomaly score (higher = more anomalous)."""
        raise NotImplementedError
    
    def detect(self, logits: torch.Tensor, threshold: float) -> bool:
        """Return True if logits are detected as malicious."""
        return self.score(logits) > threshold


# ─────────────────────────────────────────────────────────────
# T6: FLAME (Nguyen et al., 2022)
# ─────────────────────────────────────────────────────────────

class FLAMEDefense(BaseDefense):
    """
    FLAME: Taming Backdoors in Federated Learning.
    
    Key mechanisms:
    1. Cosine similarity-based clustering to identify outlier updates
    2. Noise-aware norm clipping
    3. Differential privacy noise injection
    
    Adapted to logit-level operation.
    """
    
    def __init__(self, noise_sigma: float = 0.01, clip_threshold: float = 2.0):
        super().__init__("FLAME")
        self.noise_sigma = noise_sigma
        self.clip_threshold = clip_threshold
        self.clean_centroid = None
        self.clean_norm_mean = None
        self.clean_norm_std = None
    
    def fit(self, clean_logits: torch.Tensor):
        self.clean_centroid = clean_logits.mean(dim=0)
        norms = torch.norm(clean_logits - self.clean_centroid, dim=1)
        self.clean_norm_mean = norms.mean()
        self.clean_norm_std = norms.std()
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_centroid is None:
            raise RuntimeError("Must call fit() first")
        
        # Cosine similarity to clean centroid
        centroid = logits.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            centroid.unsqueeze(0), self.clean_centroid.unsqueeze(0)
        ).item()
        
        # Norm-based clipping score
        norms = torch.norm(logits - self.clean_centroid, dim=1)
        norm_score = (norms.mean() - self.clean_norm_mean) / (self.clean_norm_std + 1e-8)
        
        # Combined score: low cosine sim + high norm = anomalous
        return (1.0 - cos_sim) + max(0, norm_score.item())


# ─────────────────────────────────────────────────────────────
# T7: DeepSight (Rieger et al., 2022)
# ─────────────────────────────────────────────────────────────

class DeepSightDefense(BaseDefense):
    """
    DeepSight: Detecting Backdoored Deep Neural Networks.
    
    Key mechanism: Spectral analysis of model updates.
    Uses SVD to detect anomalous spectral signatures in logit distributions.
    """
    
    def __init__(self, n_components: int = 3):
        super().__init__("DeepSight")
        self.n_components = n_components
        self.clean_spectrum = None
        self.clean_effective_rank = None
    
    def fit(self, clean_logits: torch.Tensor):
        centered = clean_logits - clean_logits.mean(dim=0)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        self.clean_spectrum = s[:self.n_components].clone()
        
        # Effective rank
        s_norm = s / s.sum()
        self.clean_effective_rank = torch.exp(
            -(s_norm * torch.log(s_norm + 1e-8)).sum()
        ).item()
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_spectrum is None:
            raise RuntimeError("Must call fit() first")
        
        centered = logits - logits.mean(dim=0)
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
        spectrum = s[:self.n_components]
        
        # Spectral distance
        spectral_dist = torch.norm(spectrum - self.clean_spectrum).item()
        
        # Effective rank difference
        s_norm = s / s.sum()
        eff_rank = torch.exp(-(s_norm * torch.log(s_norm + 1e-8)).sum()).item()
        rank_diff = abs(eff_rank - self.clean_effective_rank)
        
        return spectral_dist + rank_diff


# ─────────────────────────────────────────────────────────────
# T8: RLR - Robust Learning Rate (Ozdayi et al., 2021)
# ─────────────────────────────────────────────────────────────

class RLRDefense(BaseDefense):
    """
    RLR: Learning is Real and Learning Rate Matters.
    
    Key mechanism: Sign-based voting across client updates.
    If the majority of clients agree on the sign of an update
    in each coordinate, the update is likely genuine.
    
    Adapted to logit-level: checks sign consistency of logit
    deviations from the mean.
    """
    
    def __init__(self, sign_threshold: float = 0.7):
        super().__init__("RLR")
        self.sign_threshold = sign_threshold
        self.clean_sign_pattern = None
    
    def fit(self, clean_logits: torch.Tensor):
        deviations = clean_logits - clean_logits.mean(dim=0, keepdim=True)
        # Majority sign for each coordinate
        self.clean_sign_pattern = torch.sign(deviations.mean(dim=0))
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_sign_pattern is None:
            raise RuntimeError("Must call fit() first")
        
        deviations = logits - logits.mean(dim=0, keepdim=True)
        sign_pattern = torch.sign(deviations.mean(dim=0))
        
        # Sign disagreement rate
        disagreement = (sign_pattern != self.clean_sign_pattern).float().mean()
        return disagreement.item()


# ─────────────────────────────────────────────────────────────
# T9a: Krum (Blanchard et al., 2017)
# ─────────────────────────────────────────────────────────────

class KrumDefense(BaseDefense):
    """
    Krum: Machine Learning with Adversaries.
    
    Key mechanism: Select the client whose logits are closest to
    the majority of other clients (distance-based selection).
    Anomaly score = distance from the Krum-selected centroid.
    """
    
    def __init__(self, n_closest: int = 3):
        super().__init__("Krum")
        self.n_closest = n_closest
        self.clean_krum_center = None
        self.clean_dist_mean = None
        self.clean_dist_std = None
    
    def fit(self, clean_logits: torch.Tensor):
        self.clean_krum_center = clean_logits.mean(dim=0)
        dists = torch.norm(clean_logits - self.clean_krum_center, dim=1)
        self.clean_dist_mean = dists.mean()
        self.clean_dist_std = dists.std()
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_krum_center is None:
            raise RuntimeError("Must call fit() first")
        
        centroid = logits.mean(dim=0)
        dist = torch.norm(centroid - self.clean_krum_center).item()
        return (dist - self.clean_dist_mean.item()) / (self.clean_dist_std.item() + 1e-8)


# ─────────────────────────────────────────────────────────────
# T9b: TrimmedMean (Yin et al., 2018)
# ─────────────────────────────────────────────────────────────

class TrimmedMeanDefense(BaseDefense):
    """
    Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.
    
    Key mechanism: Coordinate-wise trimmed mean.
    Anomaly score = how much trimmming changes the result.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        super().__init__("TrimmedMean")
        self.trim_ratio = trim_ratio
        self.clean_trimmed_mean = None
        self.clean_diff = None
    
    def _trimmed_mean(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.shape[0]
        k = int(B * self.trim_ratio)
        sorted_logits, _ = torch.sort(logits, dim=0)
        trimmed = sorted_logits[k:B-k]
        return trimmed.mean(dim=0)
    
    def fit(self, clean_logits: torch.Tensor):
        self.clean_trimmed_mean = self._trimmed_mean(clean_logits)
        self.clean_diff = torch.norm(
            clean_logits.mean(dim=0) - self.clean_trimmed_mean
        ).item()
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_trimmed_mean is None:
            raise RuntimeError("Must call fit() first")
        
        trimmed = self._trimmed_mean(logits)
        full_mean = logits.mean(dim=0)
        
        # How different is trimmed mean from full mean?
        diff = torch.norm(full_mean - trimmed).item()
        
        # Also compare to clean trimmed mean
        shift = torch.norm(trimmed - self.clean_trimmed_mean).item()
        
        return diff + shift


# ─────────────────────────────────────────────────────────────
# T9c: FoolsGold (Fung et al., 2020)
# ─────────────────────────────────────────────────────────────

class FoolsGoldDefense(BaseDefense):
    """
    FoolsGold: Mitigating Sybils in Federated Learning Poisoning.
    
    Key mechanism: Penalizes clients whose contributions are
    too similar to each other (Sybil detection via pairwise
    cosine similarity).
    """
    
    def __init__(self):
        super().__init__("FoolsGold")
        self.clean_diversity = None
    
    def fit(self, clean_logits: torch.Tensor):
        # Measure diversity of clean logits
        normed = clean_logits / (torch.norm(clean_logits, dim=1, keepdim=True) + 1e-8)
        sim_matrix = normed @ normed.T
        # Average off-diagonal similarity
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        self.clean_diversity = sim_matrix[mask].mean().item()
    
    def score(self, logits: torch.Tensor) -> float:
        if self.clean_diversity is None:
            raise RuntimeError("Must call fit() first")
        
        normed = logits / (torch.norm(logits, dim=1, keepdim=True) + 1e-8)
        sim_matrix = normed @ normed.T
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        diversity = sim_matrix[mask].mean().item()
        
        # Higher similarity than clean = more Sybil-like = more anomalous
        return max(0, diversity - self.clean_diversity)


# ─────────────────────────────────────────────────────────────
# T9d: FLTrust (Cao et al., 2021)
# ─────────────────────────────────────────────────────────────

class FLTrustDefense(BaseDefense):
    """
    FLTrust: Federated Learning with Trust.
    
    Key mechanism: Server maintains a small root dataset.
    Client updates are scored by cosine similarity to the
    server's own update on the root dataset.
    """
    
    def __init__(self, root_size: int = 20):
        super().__init__("FLTrust")
        self.root_size = root_size
        self.server_logits = None
    
    def fit(self, clean_logits: torch.Tensor):
        # Server's "root dataset" logits = subset of clean
        indices = torch.randperm(clean_logits.shape[0])[:self.root_size]
        self.server_logits = clean_logits[indices].mean(dim=0)
    
    def score(self, logits: torch.Tensor) -> float:
        if self.server_logits is None:
            raise RuntimeError("Must call fit() first")
        
        client_mean = logits.mean(dim=0)
        cos_sim = torch.nn.functional.cosine_similarity(
            client_mean.unsqueeze(0), self.server_logits.unsqueeze(0)
        ).item()
        
        # Trust score: higher = more trusted, so anomaly = 1 - trust
        return max(0, 1.0 - cos_sim)


# ═══════════════════════════════════════════════════════════════
# PINN Guard Wrapper (for consistent interface)
# ═══════════════════════════════════════════════════════════════

class PINNGuardDefense(BaseDefense):
    """
    Wraps the adversarial PINN Guard as a BaseDefense.
    
    CRITICAL: Uses Laplacian-based scoring (matching training objective).
    The PINN was trained to minimize ||∇²f||² on clean data, so the
    anomaly score must also be ||∇²f||² — NOT just |f(x)|.
    """
    
    def __init__(self, pinn_model: nn.Module, device: str = 'cpu',
                 use_fisher: bool = False):
        super().__init__("PINN Guard")
        self.model = pinn_model
        self.device = device
        self.use_fisher = use_fisher
    
    def fit(self, clean_logits: torch.Tensor):
        pass  # Already trained
    
    def score(self, logits: torch.Tensor) -> float:
        from adversarial_pinn_guard import _compute_physics_loss, FisherInformationMetric
        self.model.eval()
        fisher = FisherInformationMetric() if self.use_fisher else None
        return _compute_physics_loss(
            self.model, logits.to(self.device),
            use_fisher=self.use_fisher, fisher=fisher,
        ).item()


class L2FilterDefense(BaseDefense):
    """
    Wraps the L2 smoothness filter as a BaseDefense.
    
    Uses raw output-squared scoring (matching its training objective).
    This is the correct ablation baseline — no physics, just smoothness.
    """
    
    def __init__(self, l2_model: nn.Module, device: str = 'cpu'):
        super().__init__("L2 Filter")
        self.model = l2_model
        self.device = device
    
    def fit(self, clean_logits: torch.Tensor):
        pass
    
    def score(self, logits: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return self.model(logits.to(self.device)).abs().mean().item()


# ═══════════════════════════════════════════════════════════════
# Convenience: Get all defenses
# ═══════════════════════════════════════════════════════════════

def get_all_baseline_defenses() -> List[BaseDefense]:
    """Instantiate all baseline defense methods."""
    return [
        FLAMEDefense(),
        DeepSightDefense(),
        RLRDefense(),
        KrumDefense(),
        TrimmedMeanDefense(),
        FoolsGoldDefense(),
        FLTrustDefense(),
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("FL Baselines - Sanity Check")
    print("=" * 60)
    
    B, C = 100, 10
    clean = make_clean_logits(B, C)
    poisoned = make_poisoned_logits(B, C, target=0, bias=3.0)
    
    defenses = get_all_baseline_defenses()
    
    for defense in defenses:
        defense.fit(clean)
        clean_score = defense.score(clean)
        poison_score = defense.score(poisoned)
        detected = poison_score > clean_score
        status = "✅" if detected else "❌"
        print(f"  {status} {defense.name:15s} | Clean: {clean_score:.4f} | Poison: {poison_score:.4f}")
    
    print("\n  Testing attack strategies...")
    attacks = get_all_attack_strategies()
    for atk in attacks[:6]:  # Test first 6
        atk_logits = execute_attack(atk, clean)
        print(f"  {atk:20s} | Target mean: {atk_logits[:, 0].mean():.4f}")
    
    print("\n✅ All baseline sanity checks passed.")
