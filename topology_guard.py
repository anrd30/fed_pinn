import numpy as np
import torch
import time
from ripser import ripser
from typing import Dict, Any, Optional

class TopologyGuard:
    """
    Structural Defense Layer for Federated Learning.
    Uses Persistent Homology (TDA) to detect 'logit shells' in point clouds.
    Functions with zero access to client model internals (Logit Cloud only).
    """
    def __init__(self, max_dim: int = 1, thresh: float = 2.0):
        self.max_dim = max_dim
        self.thresh = thresh
        self.clean_baseline_persistence = None

    def compute_persistence(self, logits: np.ndarray) -> Dict[str, Any]:
        """
        Computes the persistence diagrams for H0 and H1.
        """
        # Ensure data is 2D point cloud (N_samples, C_logits)
        if len(logits.shape) != 2:
            raise ValueError(f"Expected 2D logits array, got {logits.shape}")
        
        start_time = time.time()
        # Compute Vietoris-Rips filtration
        result = ripser(logits, maxdim=self.max_dim)
        dgms = result['dgms']
        elapsed = time.time() - start_time
        
        # H1 is the primary cyclic signature of coordinated poisoning
        h1_dgm = dgms[1]
        
        if len(h1_dgm) == 0:
            max_persistence = 0.0
            total_persistence = 0.0
        else:
            lifespans = h1_dgm[:, 1] - h1_dgm[:, 0]
            max_persistence = np.max(lifespans)
            total_persistence = np.sum(lifespans)

        return {
            'dgms': dgms,
            'max_persistence': max_persistence,
            'total_persistence': total_persistence,
            'compute_time': elapsed
        }

    def fit(self, clean_logits_batches: list):
        """
        Calculates the baseline persistence of clean batches.
        """
        baseline_scores = []
        for batch in clean_logits_batches:
            if isinstance(batch, torch.Tensor):
                batch = batch.detach().cpu().numpy()
            res = self.compute_persistence(batch)
            baseline_scores.append(res['max_persistence'])
        
        self.clean_baseline_persistence = np.mean(baseline_scores)
        print(f"Topology Guard Baseline Fit: Mean Max-Persistence = {self.clean_baseline_persistence:.6f}")

    def score(self, logits: Any) -> float:
        """
        Returns the topological outlier score.
        Higher score = higher likelihood of coordinated poisoning.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
            
        res = self.compute_persistence(logits)
        
        # We normalize by the clean baseline if available
        if self.clean_baseline_persistence is not None and self.clean_baseline_persistence > 0:
            return res['max_persistence'] / self.clean_baseline_persistence
        
        return res['max_persistence']

# ── Integration Wrapper ───────────────────────────────────────────────

from fl_baselines import BaseDefense

class TopologyGuardDefense(BaseDefense):
    def __init__(self, max_dim: int = 1):
        super().__init__(name="Topology Guard (H1 Persistence)")
        self.guard = TopologyGuard(max_dim=max_dim)

    def fit(self, clean_logits: torch.Tensor):
        # We simulate multiple batches from the clean logits to get a distribution
        B, C = clean_logits.shape
        sample_batches = []
        for _ in range(10):
            # Add small noise to simulate variability
            noisy_clean = clean_logits + torch.randn(B, C) * 0.05
            sample_batches.append(noisy_clean.cpu().numpy())
        
        self.guard.fit(sample_batches)

    def score(self, logits: torch.Tensor) -> float:
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        return self.guard.score(logits)
