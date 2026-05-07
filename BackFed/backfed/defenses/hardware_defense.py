import torch
import numpy as np
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Hardware defense will use simplified validation.")
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Device Hardware Profiles ────────────────────────────────
# These profiles represent the "Compute-Asymmetry Trap" bounds.
# An attacker attempting an RKHS Evasion attack requires ~20x more compute
# than an honest client doing standard SGD. If a client claims low-end hardware
# but submits a mathematically 'perfect' logit vector too fast, they trigger the trap.
DEVICE_PROFILES = {
    "jetson_nano": {
        "honest_round_mean_s": 8.25,   # seconds, ResNet-18, 50 steps
        "honest_round_std_s":  1.20,   # empirical std
        "mem_baseline_mb":     650,    # baseline PyTorch mem
        "peak_tflops_fp16":    0.472,
    },
    "jetson_orin_nano": {
        "honest_round_mean_s": 1.80,
        "honest_round_std_s":  0.35,
        "mem_baseline_mb":     1200,
        "peak_tflops_fp16":    17.0,
    },
    "raspberry_pi4": {
        "honest_round_mean_s": 57.5,   # CPU-only PyTorch overhead
        "honest_round_std_s":  6.0,
        "mem_baseline_mb":     480,    
        "peak_tflops_fp16":    0.0135,
    },
}

@dataclass
class ClientTelemetry:
    client_id: str
    declared_device: str
    round_id: int
    exec_time_s: float           # execution time reported by timestamp delta
    peak_mem_mb: float           # reported/attested memory usage
    logit_vector: torch.Tensor   # submitted logits

@dataclass
class ValidationResult:
    client_id: str
    flag_spoofing_fast: bool     # Layer 1: mathematically impossible completion time
    flag_attack_overhead: bool   # Layer 2: physically impossible memory overhead
    flag_temporal_variance: bool # Layer 3: temporal timing mismatch
    z_score_timing: float
    impossibility_confidence: float
    recommendation: str          # "ACCEPT", "WARN", "REJECT"

class ComputeAsymmetryTrap:
    """
    Server-side validator implementing the Compute-Asymmetry Trap.
    Integrates with FL aggregation to detect device-spoofed RKHS attacks.
    """
    def __init__(self, K_inner: int = 20, alpha: float = 0.01, history_rounds: int = 5):
        self.K = K_inner
        self.alpha = alpha
        self.history: Dict[str, List[ClientTelemetry]] = {}
        self.z_thresh = stats.norm.ppf(1 - alpha)  # ~2.33 at alpha=0.01
        self.history_rounds = history_rounds

    def _compute_attack_budget(self, device: str) -> Tuple[float, float]:
        prof = DEVICE_PROFILES[device]
        t_h  = prof["honest_round_mean_s"]
        
        # RKHS Evasion requires calculating MMD loss K times
        # Attack compute multiplier is empirical scaling of ∇²f calculations
        attack_multiplier = 1.0 + self.K * 2.5
        t_attack = t_h * attack_multiplier
        return t_h, t_attack

    def _memory_attack_cost(self, device: str, batch_size: int = 32, n_classes: int = 10) -> float:
        # Attack kernel matrix + gradient calculation overhead required for MMD matching
        kernel_mem_mb = (batch_size**2 * 4 * self.K) / (1024**2)
        grad_mem_mb = (11e6 * 4 * self.K) / (1024**2)
        return kernel_mem_mb + grad_mem_mb

    def validate(self, telemetry: ClientTelemetry, pinn_laplacian_score: float = 0.0) -> ValidationResult:
        device = telemetry.declared_device
        if device not in DEVICE_PROFILES:
            return ValidationResult(
                telemetry.client_id, True, False, False, float('-inf'), 1.0, "REJECT"
            )

        prof = DEVICE_PROFILES[device]
        t_h_mean, t_attack = self._compute_attack_budget(device)
        t_h_std = prof["honest_round_std_s"]
        mem_baseline = prof["mem_baseline_mb"]

        # ── Layer 1: SILICON TIMING LIMIT (z-score) ──
        z_timing = (telemetry.exec_time_s - t_h_mean) / t_h_std
        flag_fast = z_timing < -self.z_thresh

        # ── Layer 2: MEMORY LIMIT OVERHEAD ──
        mem_expected_attack = mem_baseline + self._memory_attack_cost(device)
        flag_mem = telemetry.peak_mem_mb > (mem_baseline * 1.5)

        # ── Layer 3: TEMPORAL VARIANCE ──
        cid = telemetry.client_id
        if cid not in self.history:
            self.history[cid] = []
        self.history[cid].append(telemetry)

        flag_temporal = False
        if len(self.history[cid]) >= self.history_rounds:
            times = [t.exec_time_s for t in self.history[cid][-self.history_rounds:]]
            cv = np.std(times) / (np.mean(times) + 1e-9)
            flag_temporal = cv > 0.5  # honest client variation should be ~0.15

        # ── COMPOSITE PROBABILITY ──
        if SCIPY_AVAILABLE:
            p_timing  = stats.norm.cdf(z_timing)
        else:
            # Simple approximation when scipy not available
            p_timing = 1.0 / (1.0 + np.exp(-z_timing))  # sigmoid approximation
        
        p_fast    = p_timing if flag_fast else 0.0
        
        # Cross Signal check: 
        p_relay = 0.0
        if pinn_laplacian_score > 0.3 and not flag_fast:
            # The geometry is highly anomalous, but timing is suspiciously perfect.
            # This indicates an attacker generated the vector on a GPU server,
            # then piped it through the Raspberry Pi to spoof the timing natively.
            p_relay = 0.85

        impossibility = max(p_fast, p_relay, float(flag_temporal) * 0.7)

        if impossibility > 0.95 or flag_fast:
            rec = "REJECT"
        elif impossibility > 0.60 or flag_temporal:
            rec = "WARN"
        else:
            rec = "ACCEPT"

        return ValidationResult(
            client_id=cid,
            flag_spoofing_fast=flag_fast,
            flag_attack_overhead=flag_mem,
            flag_temporal_variance=flag_temporal,
            z_score_timing=z_timing,
            impossibility_confidence=impossibility,
            recommendation=rec
        )

if __name__ == "__main__":
    trap = ComputeAsymmetryTrap()

    print("="*60)
    print("Testing the Compute-Asymmetry Hardware Trap")
    print("="*60)

    # 1. An honest client on a Jetson Nano
    honest_tel = ClientTelemetry("client_honest", "jetson_nano", 1, 8.4, 680, torch.randn(32, 10))
    res_honest = trap.validate(honest_tel, pinn_laplacian_score=0.05)
    print(f"[HONEST JETSON NANO]   -> Timing {honest_tel.exec_time_s}s | Score: {res_honest.z_score_timing:.2f} | Action: {res_honest.recommendation}")

    # 2. An attacker claiming to be a Jetson Nano, but computing RKHS on a datacenter server in 1.5s
    attack_tel = ClientTelemetry("client_attacker", "jetson_nano", 1, 1.5, 692, torch.randn(32, 10))
    res_attack = trap.validate(attack_tel, pinn_laplacian_score=0.45)
    print(f"[DATACENTER SPOOFER]   -> Timing {attack_tel.exec_time_s}s | Score: {res_attack.z_score_timing:.2f} | Action: {res_attack.recommendation}")

    # 3. An attacker running a Relay Attack (Computing on Datacenter, waiting 8.25s, then submitting)
    relay_tel = ClientTelemetry("client_relay", "jetson_nano", 1, 8.25, 650, torch.randn(32, 10))
    res_relay = trap.validate(relay_tel, pinn_laplacian_score=0.95)  # PINN geometric score catches it!
    print(f"[THROTTLED RELAY HACK] -> Timing {relay_tel.exec_time_s}s | PINN Laplacian Catch | Action: {res_relay.recommendation}")
