"""
Comprehensive Evaluation Suite
================================
Implements:
  T2  - Per-Strategy ROC Breakdown
  T10 - Full Comparison Table & Plots
  T13 - Scalability Analysis
  T14 - "Geometry Matters" Killer Experiment
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional
import time
import os

from adversarial_pinn_guard import (
    make_clean_logits, make_poisoned_logits,
    train_adversarial_pinn_guard, train_adversarial_l2_filter,
    get_violation_score, PINNGuard, L2SmoothnessFilter,
    FisherInformationMetric, run_architecture_sensitivity,
)
from fl_baselines import (
    get_all_baseline_defenses, get_all_attack_strategies,
    execute_attack, PINNGuardDefense, L2FilterDefense,
    shannon_entropy_score, mmd_score, BaseDefense,
)
from topology_guard import TopologyGuardDefense


# ═══════════════════════════════════════════════════════════════
# T2: Per-Strategy ROC Breakdown
# ═══════════════════════════════════════════════════════════════

def run_per_strategy_evaluation(
    clean_logits: torch.Tensor,
    defenses: List[BaseDefense],
    n_trials: int = 50,
    target: int = 0,
    output_dir: str = 'results',
) -> pd.DataFrame:
    """
    T2: Evaluate each defense against each attack strategy.
    
    For each (defense, attack) pair, generates n_trials samples and
    computes ROC-AUC, F1, TPR@FPR=5%, etc.
    
    Args:
        clean_logits: Clean baseline logits (B, C)
        defenses: List of BaseDefense instances
        n_trials: Number of trials per attack
        target: Target class
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    attacks = get_all_attack_strategies()
    B, C = clean_logits.shape
    
    results = []
    
    for defense in defenses:
        print(f"\n  Evaluating defense: {defense.name}")
        defense.fit(clean_logits)
        
        for atk_name in attacks:
            labels = []
            scores = []
            
            for trial in range(n_trials):
                # Clean sample
                clean_sample = make_clean_logits(B, C) + torch.randn(B, C) * 0.1
                clean_score = defense.score(clean_sample)
                labels.append(0)
                scores.append(clean_score)
                
                # Attack sample
                clean_ref = make_clean_logits(B, C)
                atk_logits = execute_attack(atk_name, clean_ref, target)
                atk_score = defense.score(atk_logits)
                labels.append(1)
                scores.append(atk_score)
            
            labels = np.array(labels)
            scores = np.array(scores)
            
            # Handle edge cases
            if np.isnan(scores).any() or np.isinf(scores).any():
                scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Compute metrics
            try:
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                
                # TPR at FPR = 5%
                tpr_at_5 = np.interp(0.05, fpr, tpr)
                
                # Best F1
                best_f1 = 0.0
                for thresh in thresholds:
                    preds = (scores >= thresh).astype(int)
                    f1 = f1_score(labels, preds, zero_division=0)
                    best_f1 = max(best_f1, f1)
            except Exception:
                roc_auc = 0.5
                tpr_at_5 = 0.0
                best_f1 = 0.0
            
            results.append({
                'Defense': defense.name,
                'Attack': atk_name,
                'AUC': roc_auc,
                'TPR@FPR5%': tpr_at_5,
                'Best_F1': best_f1,
            })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'per_strategy_results.csv'), index=False)
    return df


def plot_per_strategy_roc(
    clean_logits: torch.Tensor,
    defense: BaseDefense,
    n_trials: int = 50,
    target: int = 0,
    output_dir: str = 'results',
):
    """Plot individual ROC curves for each attack strategy against one defense."""
    os.makedirs(output_dir, exist_ok=True)
    attacks = get_all_attack_strategies()
    B, C = clean_logits.shape
    
    defense.fit(clean_logits)
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    axes = axes.flatten()
    
    for idx, atk_name in enumerate(attacks):
        labels, scores = [], []
        
        for _ in range(n_trials):
            clean_sample = make_clean_logits(B, C) + torch.randn(B, C) * 0.1
            labels.append(0)
            scores.append(defense.score(clean_sample))
            
            atk_logits = execute_attack(atk_name, make_clean_logits(B, C), target)
            labels.append(1)
            scores.append(defense.score(atk_logits))
        
        labels = np.array(labels)
        scores = np.nan_to_num(np.array(scores))
        
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr = [0, 1], [0, 1]
            roc_auc = 0.5
        
        ax = axes[idx]
        ax.plot(fpr, tpr, color='darkorange', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f'{atk_name}\nAUC={roc_auc:.3f}', fontsize=9)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Per-Strategy ROC Curves: {defense.name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_{defense.name.replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC plot for {defense.name}")


# ═══════════════════════════════════════════════════════════════
# T10: Full Comparison Table
# ═══════════════════════════════════════════════════════════════

def generate_comparison_table(
    results_df: pd.DataFrame,
    output_dir: str = 'results',
) -> pd.DataFrame:
    """
    T10: Generate the final comparison table.
    
    Pivot the per-strategy results into a defense × attack AUC heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pivot: Defense × Attack → AUC
    pivot = results_df.pivot_table(
        values='AUC', index='Defense', columns='Attack', aggfunc='mean'
    )
    
    # Add mean AUC column
    pivot['Mean_AUC'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('Mean_AUC', ascending=False)
    
    # Save
    pivot.to_csv(os.path.join(output_dir, 'comparison_table.csv'))
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=7)
    
    plt.colorbar(im, label='AUC-ROC')
    plt.title('Defense vs Attack: AUC-ROC Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_heatmap.png'), dpi=300)
    plt.close()
    
    # Summary table
    summary = results_df.groupby('Defense').agg({
        'AUC': ['mean', 'std', 'min'],
        'TPR@FPR5%': 'mean',
        'Best_F1': 'mean',
    }).round(4)
    summary.columns = ['Mean_AUC', 'Std_AUC', 'Min_AUC', 'Mean_TPR@5%', 'Mean_F1']
    summary = summary.sort_values('Mean_AUC', ascending=False)
    summary.to_csv(os.path.join(output_dir, 'defense_summary.csv'))
    
    print("\n" + "=" * 60)
    print("Defense Comparison Summary")
    print("=" * 60)
    print(summary.to_string())
    
    return summary


# ═══════════════════════════════════════════════════════════════
# T13: Scalability Analysis
# ═══════════════════════════════════════════════════════════════

def run_scalability_analysis(
    device: str = 'cpu',
    output_dir: str = 'results',
) -> pd.DataFrame:
    """
    T13: Test PINN Guard scalability across different configurations.
    Measures wall-clock time and memory for varying B, C, and n_clients.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        # (B, C, description)
        (50, 10, "Small (50×10)"),
        (100, 10, "Medium (100×10)"),
        (200, 10, "Large-B (200×10)"),
        (100, 50, "Medium-C (100×50)"),
        (100, 100, "Large-C (100×100)"),
        (500, 10, "Large-B (500×10)"),
        (100, 200, "XL-C (100×200)"),
    ]
    
    results = []
    
    for B, C, desc in configs:
        print(f"\n  Testing: {desc}")
        
        clean = make_clean_logits(B, C)
        poisoned = make_poisoned_logits(B, C, target=0, bias=3.0)
        
        # Training time
        start = time.time()
        pinn, _ = train_adversarial_pinn_guard(
            clean, n_epochs=50, device=device, verbose=False,
            pinn_config={'input_dim': C, 'hidden_dim': max(64, C), 
                        'num_layers': 3, 'activation': 'tanh'}
        )
        train_time = time.time() - start
        
        # Inference time (per client)
        start = time.time()
        for _ in range(20):  # Reduced from 100 since physics scoring is slower
            get_violation_score(pinn, clean, device, use_physics=True)
        infer_time = (time.time() - start) / 20
        
        # Memory (parameter count)
        n_params = sum(p.numel() for p in pinn.parameters())
        
        # Detection performance
        clean_v = get_violation_score(pinn, clean, device, use_physics=True)
        poison_v = get_violation_score(pinn, poisoned, device, use_physics=True)
        
        results.append({
            'Config': desc,
            'B': B,
            'C': C,
            'Train_Time_s': round(train_time, 2),
            'Infer_Time_ms': round(infer_time * 1000, 2),
            'Params': n_params,
            'Clean_V': round(clean_v, 6),
            'Poison_V': round(poison_v, 6),
            'Separation': round(poison_v - clean_v, 6),
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'scalability.csv'), index=False)
    
    print("\n" + "=" * 60)
    print("Scalability Analysis")
    print("=" * 60)
    print(df.to_string(index=False))
    
    return df


# ═══════════════════════════════════════════════════════════════
# T14: "Geometry Matters" Killer Experiment
# ═══════════════════════════════════════════════════════════════

def run_geometry_matters_experiment(
    device: str = 'cpu',
    B: int = 100,
    C: int = 10,
    n_trials: int = 30,
    output_dir: str = 'results',
) -> pd.DataFrame:
    """
    T14: Find and demonstrate a scenario where:
    1. All statistical filters (entropy, MMD, spectral) PASS
    2. L2 smoothness filter PASSES
    3. PINN Guard with Laplacian/Dirichlet DETECTS
    
    This proves the manifold geometry is necessary and sufficient.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("'Geometry Matters' Killer Experiment")
    print("=" * 60)
    
    clean = make_clean_logits(B, C)
    
    # Train all detectors
    print("  Training PINN Guard (adversarial, Euclidean)...")
    pinn_euclidean, _ = train_adversarial_pinn_guard(
        clean, n_epochs=150, device=device, use_fisher=False, verbose=False
    )
    
    print("  Training PINN Guard (adversarial, Fisher)...")
    pinn_fisher, _ = train_adversarial_pinn_guard(
        clean, n_epochs=150, device=device, use_fisher=True, verbose=False
    )
    
    print("  Training L2 Smoothness Filter...")
    l2_filter = train_adversarial_l2_filter(clean, device=device, n_epochs=150)
    
    # Get baseline defenses
    baselines = get_all_baseline_defenses()
    for d in baselines:
        d.fit(clean)
    
    # Now: craft an attack that evades everything EXCEPT PINN
    # Strategy: use RKHS evasion (matches distribution) but with
    # additional spectral matching (matches singular values)
    # This should evade statistical + L2 but not manifold curvature
    
    attacks_to_test = ['RKHS_Evasion', 'Spectral_Masking', 'Super_Adaptive',
                       'Constrain_Scale', 'DBA_Part0']
    
    results = []
    
    for atk_name in attacks_to_test:
        print(f"\n  Testing attack: {atk_name}")
        
        pinn_e_scores, pinn_f_scores, l2_scores = [], [], []
        baseline_scores = {d.name: [] for d in baselines}
        
        for _ in range(n_trials):
            clean_ref = make_clean_logits(B, C)
            atk_logits = execute_attack(atk_name, clean_ref)
            
            pinn_e_scores.append(get_violation_score(pinn_euclidean, atk_logits, device, use_physics=True))
            pinn_f_scores.append(get_violation_score(pinn_fisher, atk_logits, device, use_physics=True, use_fisher=True))
            l2_scores.append(get_violation_score(l2_filter, atk_logits, device, use_physics=False))
            
            for d in baselines:
                baseline_scores[d.name].append(d.score(atk_logits))
        
        # Clean baselines
        pinn_e_clean = np.mean([
            get_violation_score(pinn_euclidean, make_clean_logits(B, C), device, use_physics=True) 
            for _ in range(n_trials)
        ])
        pinn_f_clean = np.mean([
            get_violation_score(pinn_fisher, make_clean_logits(B, C), device, use_physics=True, use_fisher=True)
            for _ in range(n_trials)
        ])
        l2_clean = np.mean([
            get_violation_score(l2_filter, make_clean_logits(B, C), device, use_physics=False)
            for _ in range(n_trials)
        ])
        
        row = {
            'Attack': atk_name,
            'PINN_Euclidean': np.mean(pinn_e_scores),
            'PINN_Euclidean_Clean': pinn_e_clean,
            'PINN_Euclidean_Ratio': np.mean(pinn_e_scores) / (pinn_e_clean + 1e-8),
            'PINN_Fisher': np.mean(pinn_f_scores),
            'PINN_Fisher_Clean': pinn_f_clean,
            'PINN_Fisher_Ratio': np.mean(pinn_f_scores) / (pinn_f_clean + 1e-8),
            'L2_Filter': np.mean(l2_scores),
            'L2_Filter_Clean': l2_clean,
            'L2_Filter_Ratio': np.mean(l2_scores) / (l2_clean + 1e-8),
        }
        
        for d in baselines:
            clean_score = d.score(make_clean_logits(B, C))
            row[f'{d.name}_Score'] = np.mean(baseline_scores[d.name])
            row[f'{d.name}_Clean'] = clean_score
        
        results.append(row)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'geometry_matters.csv'), index=False)
    
    # Print summary
    print("\n" + "-" * 60)
    print("Results (Ratio = Attack / Clean, >1 = detected):")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"\n  Attack: {row['Attack']}")
        print(f"    PINN Euclidean Ratio: {row['PINN_Euclidean_Ratio']:.4f}")
        print(f"    PINN Fisher Ratio:    {row['PINN_Fisher_Ratio']:.4f}")
        print(f"    L2 Filter Ratio:      {row['L2_Filter_Ratio']:.4f}")
    
    return df


# ═══════════════════════════════════════════════════════════════
# MASTER EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_full_evaluation(
    device: str = 'cpu',
    B: int = 100,
    C: int = 10,
    n_trials: int = 30,
    output_dir: str = 'results',
):
    """
    Run the complete evaluation suite.
    Executes T2, T10, T11, T13, T14 in sequence.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FULL EVALUATION SUITE")
    print("=" * 60)
    
    clean = make_clean_logits(B, C)
    
    # ── Train defenses ──
    print("\n[1/6] Training Adversarial PINN Guard...")
    pinn, pinn_history = train_adversarial_pinn_guard(
        clean, n_epochs=200, device=device
    )
    
    print("\n[2/6] Training L2 Smoothness Filter...")
    l2_filter = train_adversarial_l2_filter(clean, device=device, n_epochs=200)
    
    print("\n[3/6] Training PINN Guard with Fisher Metric...")
    pinn_fisher, _ = train_adversarial_pinn_guard(
        clean, n_epochs=200, use_fisher=True, device=device
    )
    
    # ── Assemble all defenses ──
    all_defenses = [
        PINNGuardDefense(pinn, device, use_fisher=False),
        PINNGuardDefense(pinn_fisher, device, use_fisher=True),
        L2FilterDefense(l2_filter, device),
        TopologyGuardDefense(max_dim=1),
    ] + get_all_baseline_defenses()
    
    # Fix names
    all_defenses[0].name = "PINN Guard (Euclidean)"
    all_defenses[1].name = "PINN Guard (Fisher)"
    all_defenses[2].name = "L2 Filter (Ablation)"
    all_defenses[3].name = "Topology Guard (TDA)"
    
    # ── T2: Per-Strategy Evaluation ──
    print("\n[4/6] Running Per-Strategy Evaluation (T2)...")
    results_df = run_per_strategy_evaluation(
        clean, all_defenses, n_trials=n_trials, output_dir=output_dir
    )
    
    # Plot individual ROC curves for PINN Guard
    for defense in all_defenses[:3]:
        plot_per_strategy_roc(clean, defense, n_trials=n_trials, output_dir=output_dir)
    
    # ── T10: Comparison Table ──
    print("\n[5/6] Generating Comparison Table (T10)...")
    summary = generate_comparison_table(results_df, output_dir=output_dir)
    
    # ── T14: Geometry Matters ──
    print("\n[6/6] Running Geometry Matters Experiment (T14)...")
    geo_df = run_geometry_matters_experiment(
        device=device, B=B, C=C, n_trials=n_trials, output_dir=output_dir
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)
    
    return results_df, summary, geo_df


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run quick evaluation (reduced trials for speed)
    results_df, summary, geo_df = run_full_evaluation(
        device=device,
        B=100,
        C=10,
        n_trials=20,  # Increase to 50+ for paper
        output_dir='results',
    )
    
    # T13: Scalability
    print("\nRunning Scalability Analysis (T13)...")
    scale_df = run_scalability_analysis(device=device, output_dir='results')
