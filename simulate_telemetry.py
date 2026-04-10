import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from hardware_defense import ComputeAsymmetryTrap, ClientTelemetry, DEVICE_PROFILES

def run_simulation(n_clients=100, n_rounds=30, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    trap = ComputeAsymmetryTrap(K_inner=20, alpha=0.01)

    results = []
    labels = []
    scores = []

    print("=" * 60)
    print("Running Hardware-Aware Telemetry Simulation")
    print("=" * 60)

    # We mainly focus on Jetson Nano logic for the simulation
    device_name = "jetson_nano"
    prof = DEVICE_PROFILES[device_name]
    t_h = prof["honest_round_mean_s"]
    t_h_std = prof["honest_round_std_s"]
    mem_h = prof["mem_baseline_mb"]
    mem_attack_extra = trap._memory_attack_cost(device_name)
    mem_attack = mem_h + mem_attack_extra

    # ── Assign Identities ──
    # 70% Honest
    # 15% Naive Datacenter Spoofers (runs fast physically)
    # 15% Smart Relay Attackers (throttles to match time, caught by PINN + Memory checks)
    identities = []
    np.random.seed(42)  # For reproducible perfect paper graphs
    for i in range(n_clients):
        r = np.random.rand()
        if r < 0.70:
            identities.append("HONEST")
        elif r < 0.85:
            identities.append("SPOOFER")
        else:
            identities.append("RELAY")

    print(f"Simulating {n_rounds} rounds of FL across {n_clients} Edge Clients...")

    for round_id in range(1, n_rounds + 1):
        for cid_idx, identity in enumerate(identities):
            client_id = f"client_{cid_idx:03d}"
            
            # Stochastic generation natively anchored to the empirical hardware profiles
            if identity == "HONEST":
                # Honest timing + noise, honest memory, low PINN geometrical anomaly score
                ex_time = max(t_h * 0.1, np.random.normal(t_h, t_h_std))
                peak_mem = np.random.normal(mem_h, mem_h * 0.05)
                pinn_score = np.clip(np.random.normal(0.1, 0.05), 0, 1)
                is_attack = 0

            elif identity == "SPOOFER":
                # A100 Datacenter: ~1.5s execution, normal memory, perfect geometric evasion (PINN=low)
                ex_time = max(0.1, np.random.normal(1.5, 0.2))
                peak_mem = np.random.normal(mem_h, mem_h * 0.05)
                pinn_score = np.clip(np.random.normal(0.05, 0.02), 0, 1) 
                is_attack = 1

            elif identity == "RELAY":
                # Smart attacker: holds the update on the proxy to perfectly match honest timing
                # but adds heavy memory context from the graph, or fails the PINN completely
                ex_time = max(t_h * 0.1, np.random.normal(t_h, t_h_std * 0.1)) # artificially clean spoofed timing
                peak_mem = np.random.normal(mem_attack, mem_attack * 0.02)
                pinn_score = np.clip(np.random.normal(0.85, 0.1), 0, 1) # MMD failed to fully mask curvature
                is_attack = 1

            # Construct Telemetry Object
            tel = ClientTelemetry(
                client_id=client_id,
                declared_device=device_name,
                round_id=round_id,
                exec_time_s=ex_time,
                peak_mem_mb=peak_mem,
                logit_vector=torch.zeros(1) # mock vector, logic only uses timing/memory
            )

            # Validate!
            res = trap.validate(tel, pinn_laplacian_score=pinn_score)

            results.append({
                "round_id": round_id,
                "client_id": client_id,
                "identity": identity,
                "exec_time_s": ex_time,
                "peak_mem_mb": peak_mem,
                "pinn_score": pinn_score,
                "impossibility": res.impossibility_confidence,
                "action": res.recommendation
            })

            labels.append(is_attack)
            scores.append(res.impossibility_confidence)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "telemetry_simulation.csv"), index=False)

    # ── Compute Hardware Validation AUC ──
    fpr, tpr, thresh = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    tpr_at_1percent = np.interp(0.01, fpr, tpr)

    print("\n" + "="*30)
    print("RESULTS SUMMARY:")
    print("="*30)
    print(f"Total Updates Trapped: {len(labels)}")
    print(f"Hardware ROC-AUC Score: {roc_auc:.4f}")
    print(f"TPR at 1% FPR:        {tpr_at_1percent:.4f}")

    # ── Generate Paper Distribution Plot ──
    honest_times = df[df["identity"] == "HONEST"]["exec_time_s"]
    spoofer_times = df[df["identity"] == "SPOOFER"]["exec_time_s"]

    plt.figure(figsize=(12, 6))
    plt.hist(honest_times, bins=40, alpha=0.7, label='Honest Edge Clients (Jetson Nano)', color='#2e86ab')
    plt.hist(spoofer_times, bins=20, alpha=0.7, label='Datacenter Spoofers (A100)', color='#d62828')
    plt.axvline(x=t_h * 0.5, color='black', linestyle='dashed', linewidth=2, label=f'Impossibility Boundary (Layer 1 Trap)')

    plt.title('Execution Time Distributions (Hardware-Aware Compute-Asymmetry)', fontsize=15, pad=15)
    plt.xlabel('Execution Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    # Shade the impossible region red
    plt.axvspan(-10, t_h * 0.5, facecolor='red', alpha=0.1)

    plt.xlim(max(0, spoofer_times.min() - 0.5), honest_times.max() + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hardware_time_distribution.png"), dpi=600) # High Res for Paper
    plt.close()
    
    print("\n[+] Distribution Plot successfully saved to 'results/hardware_time_distribution.png'")
    
if __name__ == "__main__":
    run_simulation()
